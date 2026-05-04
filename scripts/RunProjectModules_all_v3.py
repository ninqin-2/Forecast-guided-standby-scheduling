import os
import gc
import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import nbinom

import os
import gc
import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import nbinom
from numpy.polynomial.hermite import hermgauss

# =========================================================
# 全局缓存
# =========================================================
_GH_CACHE = {}
_PRECOMP_CACHE = {}


def _get_gh_nodes(K):
    """缓存 GH 节点，避免重复 hermgauss(K)"""
    if K not in _GH_CACHE:
        x, w = hermgauss(K)
        _GH_CACHE[K] = (x, w)
    return _GH_CACHE[K]


def _make_cache_key(collect, TrainDist_df, windows, K, m_max):
    """
    用对象 id 做轻量缓存键。
    适用于 ProjectModulesPlan 内部反复调用 MyErrorDistDeal 的场景。
    """
    collect_ids = tuple(id(df) for df in collect)
    key = (
        collect_ids,
        id(TrainDist_df),
        tuple(windows) if windows is not None else None,
        int(K),
        int(m_max),
        len(collect),
        len(TrainDist_df),
    )
    return key


def _first_geq_each_row(ft_table, target):
    """
    ft_table: shape (T, M+1)
    返回每行第一个 >= target 的列下标
    """
    mask = ft_table >= target
    has_hit = mask.any(axis=1)
    idx = np.argmax(mask, axis=1)   # 若全 False 会返回 0，所以要配合 has_hit
    idx = idx.astype(np.int32)
    idx[~has_hit] = ft_table.shape[1] - 1
    return idx, has_hit


def _precompute_ft_table_by_hour(r_array, p_array, hour_array,
                                 hour_param, global_mu, global_sigma,
                                 m_max=400, K=25, chunk_size=256):
    """
    一次性预计算每个样本 t 在 m=0..m_max 上的 Ft(m) 曲线。
    返回：
      ft_table: shape (T, m_max+1)
    """
    x, w = _get_gh_nodes(K)
    weights = w / np.sqrt(np.pi)
    z = np.sqrt(2.0) * x
    m_grid = np.arange(m_max + 1, dtype=np.float64)

    T = len(r_array)
    ft_table = np.empty((T, m_max + 1), dtype=np.float32)

    unique_hours = np.unique(hour_array)

    for h in unique_hours:
        idx = np.where(hour_array == h)[0]
        if idx.size == 0:
            continue

        mu_err, sigma_err, _ = hour_param.get(int(h), (global_mu, global_sigma, 0))
        sigma_err = max(float(sigma_err), 1e-6)

        # 该小时对应的误差节点
        e = mu_err + sigma_err * z  # shape (K,)

        # k_mat 只由 hour 决定，和每一行 r,p 无关，可以复用
        k_mat = np.floor(m_grid[:, None] - e[None, :]).astype(np.int64)   # (M+1, K)
        valid_mask = (k_mat >= 0)
        k_eval = np.where(valid_mask, k_mat, 0)

        # 分块，避免一次性占太大内存
        for start in range(0, idx.size, chunk_size):
            part = idx[start:start + chunk_size]
            r_chunk = r_array[part].astype(np.float64)[:, None, None]   # (B,1,1)
            p_chunk = p_array[part].astype(np.float64)[:, None, None]   # (B,1,1)

            # 广播后得到 (B, M+1, K)
            cdf_vals = nbinom.cdf(k_eval[None, :, :], r_chunk, p_chunk)
            cdf_vals *= valid_mask[None, :, :]

            # 对 K 做加权平均 -> (B, M+1)
            ft_chunk = np.tensordot(cdf_vals, weights, axes=([2], [0]))
            ft_table[part, :] = ft_chunk.astype(np.float32)

    return ft_table


def _prepare_precomputed_station_data(collect, TrainDist_df, windows, m_max=400, K=25):
    """
    把与 eps 无关的量全部预计算并缓存。
    """
    cache_key = _make_cache_key(collect, TrainDist_df, windows, K, m_max)
    if cache_key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[cache_key]

    # 1) merge 只做一次
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="inner"), collect)

    # 2) 规则应用只做一次
    out_base = apply_preemptive_rules_multi(
        merged_df, windows=windows, up_th=2, down_th=10, apply_down=True, priority="up"
    )
    out_base["Error"] = out_base["true_0"] - out_base["t_0"]
    out_base["Error_after"] = out_base["true_0"] - out_base["m_after"]

    # 3) 按小时残差拟合只做一次
    global_mu, global_sigma, hour_param = fit_residual_by_hour(TrainDist_df, shrink_strength=200)

    # 4) 与 eps 无关的参数只做一次
    mu_pred = out_base["m_after"].to_numpy(dtype=np.float64)
    r_array = out_base["Pred_Alpha"].to_numpy(dtype=np.float64)
    p_array = r_array / (r_array + mu_pred)
    hour_array = out_base["Hour"].to_numpy(dtype=np.int16)

    # 5) 一次性预计算 Ft 表
    ft_table = _precompute_ft_table_by_hour(
        r_array=r_array,
        p_array=p_array,
        hour_array=hour_array,
        hour_param=hour_param,
        global_mu=global_mu,
        global_sigma=global_sigma,
        m_max=m_max,
        K=K,
        chunk_size=256
    )

    payload = {
        "out_base": out_base,
        "global_mu": global_mu,
        "global_sigma": global_sigma,
        "hour_param": hour_param,
        "r_array": r_array,
        "p_array": p_array,
        "hour_array": hour_array,
        "ft_table": ft_table,
        "m_max": m_max,
        "K": K,
    }
    _PRECOMP_CACHE[cache_key] = payload
    return payload


def Ft_monte_carlo(m: int, r: float, p: float, mu: float, sigma: float,
                   n_mc: int = 200000, seed: int = 0) -> float:
    """
    估计 Ft(m) = P( D_hat + E <= m ) = E[ F_hat(m - E) ]
    其中 D_hat ~ NB(r, p), E ~ N(mu, sigma^2), 独立

    NB 参数化说明（scipy.stats.nbinom）：
      nbinom(n=r, p=p) 的随机变量K表示“在得到 r 次成功前失败次数”，
      K ∈ {0,1,2,...}
      CDF: P(K <= k)

    m: 你要评估的模块数阈值（整数）
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(loc=mu, scale=sigma, size=n_mc)

    # 需要计算 F_hat(m - e)
    # 由于 NB 的自变量必须是整数 k >= 0，这里采用 floor 将阈值映射到整数
    # 解释：事件 {D_hat <= m - e} 中 D_hat 为整数，因此等价于 {D_hat <= floor(m - e)}
    k = np.floor(m - e).astype(np.int64)

    # k < 0 时 CDF = 0
    out = np.zeros_like(k, dtype=np.float64)
    mask = k >= 0
    if np.any(mask):
        out[mask] = nbinom.cdf(k[mask], r, p)

    return float(out.mean())


def Ft_gh(m, r, p, mu_err, sigma_err, x, w):
    """
    Gauss-Hermite 近似 Ft(m) = E[ NB_CDF(floor(m - E)) ]
    E ~ N(mu_err, sigma_err^2)
    x, w: hermgauss(K) 返回的节点与权重
    """
    # 将标准正态的期望转换成 GH 形式：Z = sqrt(2)*x_i
    z = np.sqrt(2.0) * x
    e = mu_err + sigma_err * z

    k = np.floor(m - e).astype(np.int64)
    vals = np.zeros_like(k, dtype=float)
    mask = k >= 0
    if np.any(mask):
        vals[mask] = nbinom.cdf(k[mask], r, p)

    # E[f(Z)] ≈ (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
    return float((w @ vals) / np.sqrt(np.pi))


def find_m_vectorized_gh_hour(
    r_array, p_array, hour_array,
    hour_param, global_mu, global_sigma,
    eps=0.05, m_max=400, K=25
):
    """
    保持原函数名和输入不变。
    改为二分搜索，避免每个样本对 m=0..m_max 线性扫描。
    """
    x, w = _get_gh_nodes(K)
    T = len(r_array)

    m_star = np.zeros(T, dtype=np.int32)
    ft_star = np.zeros(T, dtype=np.float64)
    target = 1.0 - eps

    # 按小时分组，减少重复取 hour 参数
    unique_hours = np.unique(hour_array)

    for h in unique_hours:
        idx = np.where(hour_array == h)[0]
        mu_err, sigma_err, _ = hour_param.get(int(h), (global_mu, global_sigma, 0))
        sigma_err = max(float(sigma_err), 1e-6)

        for t in idx:
            r = float(r_array[t])
            p = float(p_array[t])

            lo, hi = 0, m_max
            best_m = m_max
            best_ft = Ft_gh(m_max, r, p, mu_err, sigma_err, x, w)

            while lo <= hi:
                mid = (lo + hi) // 2
                ft = Ft_gh(mid, r, p, mu_err, sigma_err, x, w)
                if ft >= target:
                    best_m = mid
                    best_ft = ft
                    hi = mid - 1
                else:
                    lo = mid + 1

            m_star[t] = best_m
            ft_star[t] = best_ft

    return m_star, ft_star

def in_hour_window(hour: pd.Series, start: int, end: int) -> pd.Series:
    """
    判断 hour 是否落在 [start, end) 区间（跨午夜支持）。
    例：(22,2) 表示 22:00-02:00；(11,13) 表示 11:00-13:00
    """
    hour = hour.astype(int)
    if start <= end:
        return (hour >= start) & (hour < end)
    else:
        return (hour >= start) | (hour < end)


def apply_preemptive_rules_multi(
    df: pd.DataFrame,
    windows=[(22, 2), (11, 13)],
    up_th: float = 10,
    down_th: float = 10,
    apply_down: bool = True,     # 是否启用“延迟降下”规则
    priority: str = "up",        # "up"：上升优先；"down"：下降优先
) -> pd.DataFrame:
    """
    在 windows 覆盖的时段内：
      - 若 max(t+1,t+2) - t >= up_th:  令 m_after 对标 max(t+1,t+2)（提前拉起）
      - 若 t - min(t+1,t+2) >= down_th: 令 m_after 对标 min(t+1,t+2)（延迟降下，可选）
    """
    df = df.copy()

    # Hour
    if "Hour" not in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df["Hour"] = df["Time"].dt.hour

    t0 = df["t_0"].astype(float)
    t1 = df["t_1"].astype(float)
    t2 = df["t_2"].astype(float)

    future_max = np.maximum(t1, t2)
    future_min = np.minimum(t1, t2)

    df["delta_up"] = future_max - t0
    df["delta_down"] = t0 - future_min

    # 初始化
    df["is_peak_any"] = False
    df["action"] = "不动作"
    df["delta_modules"] = 0.0

    # 总高峰掩码（由 windows 决定）
    for start, end in windows:
        df["is_peak_any"] |= in_hour_window(df["Hour"], start, end)

    # 触发条件（只由 windows 控制）
    up_trigger = df["is_peak_any"] & (df["delta_up"] >= up_th)
    down_trigger = df["is_peak_any"] & (df["delta_down"] >= down_th) if apply_down else pd.Series(False, index=df.index)

    # 处理优先级：避免同一行既 up 又 down 时覆盖混乱
    if priority == "up":
        # 先上升，再下降（下降只在非上升行生效）
        df.loc[up_trigger, "action"] = "提前拉起至未来峰值"
        df.loc[up_trigger, "delta_modules"] = future_max.loc[up_trigger] - t0.loc[up_trigger]


    elif priority == "down":
        # 先下降，再上升（上升只在非下降行生效）
        df.loc[down_trigger, "action"] = "延迟降下至未来低谷"
        df.loc[down_trigger, "delta_modules"] = future_min.loc[down_trigger] - t0.loc[down_trigger]

        up_eff = up_trigger & ~down_trigger
        df.loc[up_eff, "action"] = "提前拉起至未来峰值"
        df.loc[up_eff, "delta_modules"] = future_max.loc[up_eff] - t0.loc[up_eff]
    else:
        raise ValueError("priority must be 'up' or 'down'")

    # 输出模块数
    df["delta_modules_int"] = np.rint(df["delta_modules"]).astype(int)
    df["m_plan"] = t0
    df["m_after"] = (df["m_plan"] + df["delta_modules_int"]).clip(lower=0)

    return df

def compute_error_statistics(true, pred, optimized_pred=None, risk_threshold=0):
    true = np.asarray(true)
    pred = np.asarray(pred)

    error = true - pred
    error_optimized = true - optimized_pred if optimized_pred is not None else None

    error_stats = {
        "90th_percentile": np.percentile(error, 90),
        "95th_percentile": np.percentile(error, 95),
        "99th_percentile": np.percentile(error, 99),
        "max_error": np.max(error),

        # 服务降级风险：低估超过阈值的比例
        "degradation_risk": np.mean(error > risk_threshold)
    }

    if optimized_pred is not None:
        optimized_pred = np.asarray(optimized_pred)
        error_optimized = true - optimized_pred

        error_optimized_stats = {
            "90th_percentile_optimized": np.percentile(error_optimized, 90),
            "95th_percentile_optimized": np.percentile(error_optimized, 95),
            "99th_percentile_optimized": np.percentile(error_optimized, 99),
            "max_error_optimized": np.max(error_optimized),

            "degradation_risk_optimized": np.mean(error_optimized > risk_threshold)
        }
        error_stats.update(error_optimized_stats)

    return error_stats

def fit_residual_by_hour(df, shrink_strength=200):
    """
    返回：
      global_mu, global_sigma
      hour_param: dict{hour: (mu_h, sigma_h, n_h)}
    """
    tmp = df.copy()
    tmp["resid"] = tmp["True"] - tmp["Pred_Mu"]
    tmp = tmp[np.isfinite(tmp["resid"])]

    resid_all = tmp["resid"].to_numpy()
    global_mu = float(resid_all.mean())
    global_sigma = float(resid_all.std(ddof=1)) if resid_all.size > 1 else 1.0

    stats = tmp.groupby("Hour")["resid"].agg(["count", "mean", "std"]).reset_index()

    hour_param = {}
    for _, row in stats.iterrows():
        h = int(row["Hour"])
        n = int(row["count"])
        mu = float(row["mean"])
        sigma = float(row["std"]) if np.isfinite(row["std"]) and n >= 2 else global_sigma

        # shrink toward global (防止某些 hour 样本少导致 sigma 乱跳)
        w = n / (n + shrink_strength)
        mu = w * mu + (1 - w) * global_mu
        sigma = w * sigma + (1 - w) * global_sigma

        hour_param[h] = (mu, sigma, n)

    return global_mu, global_sigma, hour_param

def MyErrorDistDeal(collect, TrainDist_df, eps, state, windows=None):

    if windows is None:
        windows = [(21, 0), (11, 18)]

    # state=False 时，只返回基础规则处理结果
    if not state:
        pre = _prepare_precomputed_station_data(
            collect=collect,
            TrainDist_df=TrainDist_df,
            windows=windows,
            m_max=400,
            K=25
        )
        return pre["out_base"].copy()

    # state=True 时，直接从预计算 Ft 表上按 eps 取 m_final
    pre = _prepare_precomputed_station_data(
        collect=collect,
        TrainDist_df=TrainDist_df,
        windows=windows,
        m_max=400,
        K=25
    )

    out_df = pre["out_base"].copy()
    ft_table = pre["ft_table"]

    target = 1.0 - eps
    m_star, has_hit = _first_geq_each_row(ft_table, target)

    # 取对应 Ft 值
    row_idx = np.arange(ft_table.shape[0])
    ft_star = ft_table[row_idx, m_star]

    out_df["m_final"] = m_star.astype(int)
    out_df["Ft_final"] = ft_star
    out_df["Error_final"] = out_df["true_0"] - out_df["m_final"]

    return out_df

def ProjectModulesPlan(stat_codes, TrainDist_path):
    TrainDist_df = pd.read_csv(TrainDist_path)

    collect = []

    for k in [0, 1, 2]:
        csv_path = rf"PredictResult\reg_patchTST_1h_{stat_codes}_t{k}.csv"
        df = pd.read_csv(csv_path)
        df["Time"] = pd.to_datetime(df["Time"])

        if k == 0:
            temp = df.copy()
            temp = temp.rename(columns={"True": f"true_{k}", "Pred_Mu": f"t_{k}"})
        else:
            temp = df[["Time", "True", "Pred_Mu"]].copy()
            temp = temp.rename(columns={"True": f"true_{k}", "Pred_Mu": f"t_{k}"})

        collect.append(temp)

    target_risk_levels = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    eps_candidates = np.round(np.arange(0.01, 1.01, 0.01), 2)

    save_dir = r"PredictResult_eps"
    os.makedirs(save_dir, exist_ok=True)

    pre = _prepare_precomputed_station_data(
        collect=collect,
        TrainDist_df=TrainDist_df,
        windows=[(21, 0), (11, 18)],
        m_max=400,
        K=25
    )

    out_base = pre["out_base"]
    ft_table = pre["ft_table"]

    true_values = out_base["true_0"].to_numpy()
    pred_values = out_base["t_0"].to_numpy()

    summary_records = []

    for eps in eps_candidates:
        try:
            target = 1.0 - eps
            m_star, has_hit = _first_geq_each_row(ft_table, target)
            optimized_pred_values = m_star

            error_statistics = compute_error_statistics(
                true_values, pred_values, optimized_pred_values
            )

            degradation_risk_optimized = error_statistics.get("degradation_risk_optimized", None)

            summary_records.append({
                "eps": eps,
                "degradation_risk_optimized": degradation_risk_optimized,
                "90th_percentile_optimized": error_statistics.get("90th_percentile_optimized"),
                "95th_percentile_optimized": error_statistics.get("95th_percentile_optimized"),
                "99th_percentile_optimized": error_statistics.get("99th_percentile_optimized"),
                "max_error_optimized": error_statistics.get("max_error_optimized"),
            })

            print(f"eps={eps:.2f}, degradation_risk_optimized={degradation_risk_optimized}")

        except Exception as e:
            print(f"eps={eps:.2f} 运行失败: {e}")

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(save_dir, f"risk_summary_{stat_codes}.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"风险汇总已保存: {summary_path}")

    selected_records = []
    selected_eps_set = set()

    valid_df = summary_df.dropna(subset=["degradation_risk_optimized"]).copy()

    for target in target_risk_levels:
        if valid_df.empty:
            continue

        tmp = valid_df.copy()
        tmp["risk_gap"] = (tmp["degradation_risk_optimized"] - target).abs()
        best_row = tmp.loc[tmp["risk_gap"].idxmin()]

        selected_eps = float(best_row["eps"])
        actual_risk = float(best_row["degradation_risk_optimized"])
        risk_gap = float(best_row["risk_gap"])

        selected_eps_set.add(selected_eps)
        selected_records.append({
            "target_risk": target,
            "selected_eps": selected_eps,
            "actual_risk": actual_risk,
            "risk_gap": risk_gap,
        })

    selected_df = pd.DataFrame(selected_records)
    selected_map_path = os.path.join(save_dir, f"selected_risk_mapping_{stat_codes}.xlsx")
    selected_df.to_excel(selected_map_path, index=False)
    print(f"目标风险映射已保存: {selected_map_path}")

    saved_files = []

    for eps in sorted(selected_eps_set):
        try:
            target = 1.0 - eps
            m_star, has_hit = _first_geq_each_row(ft_table, target)
            optimized_pred_values = m_star

            error_statistics = compute_error_statistics(
                true_values, pred_values, optimized_pred_values
            )
            actual_risk = error_statistics.get("degradation_risk_optimized", None)

            out_df = out_base.copy()
            out_df["m_final"] = optimized_pred_values.astype(int)
            out_df["Error_final"] = out_df["true_0"] - out_df["m_final"]
            out_df["Ft_final"] = ft_table[np.arange(ft_table.shape[0]), optimized_pred_values]

            save_path = os.path.join(
                save_dir,
                f"reg_patchTST_1h_{stat_codes}_eps_{eps:.2f}_risk_{actual_risk:.4f}.xlsx"
            )
            out_df.to_excel(save_path, index=False)
            saved_files.append(save_path)
            print(f"已保存: {save_path}")

            del out_df, optimized_pred_values, error_statistics
            gc.collect()

        except Exception as e:
            print(f"eps={eps:.2f} 保存失败: {e}")

    return {
        "stat_codes": str(stat_codes),
        "summary_path": summary_path,
        "selected_map_path": selected_map_path,
        "saved_files": saved_files,
        "selected_df": selected_df,
        "summary_df": summary_df
    }


from datetime import datetime


def load_run_status(record_path):
    """
    读取运行状态表；不存在则创建空表
    """
    if os.path.exists(record_path):
        df = pd.read_csv(record_path, dtype={"stat_codes": str})
    else:
        df = pd.DataFrame(columns=[
            "stat_codes",
            "status",
            "last_update_time",
            "summary_path",
            "selected_map_path",
            "output_files",
            "selected_eps",
            "target_risks",
            "actual_risks",
            "message"
        ])
    return df


def save_run_status(df, record_path):
    df.to_csv(record_path, index=False, encoding="utf-8-sig")


def get_station_status(df, stat_codes):
    """
    返回某站点当前状态；若不存在返回 None
    """
    stat_codes = str(stat_codes)
    sub = df[df["stat_codes"] == stat_codes]
    if sub.empty:
        return None
    return sub.iloc[-1]["status"]


def upsert_station_status(df, record):
    """
    按 stat_codes 覆盖写入状态记录
    """
    stat_codes = str(record["stat_codes"])
    df = df[df["stat_codes"] != stat_codes].copy()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    return df


if __name__ == "__main__":
    base_files = r"F:\代码库\TLD_modular_predict\Station_DATA"
    stat_file = r"F:\代码库\TLD_modular_predict\station_324_list.xlsx"
    codes_list = r"ExamRecords.xlsx"

    # 运行状态记录表
    run_record_path = r"run_status_records.csv"

    stat_codes_df = pd.read_excel(codes_list)
    stat_codes_df["stat_codes"] = stat_codes_df["stat_codes"].astype(str)

    stat_df = pd.read_excel(stat_file)
    stat_df["电站编号"] = stat_df["电站编号"].astype(str)

    error_factor_dir = r"F:\代码库\PredictModulesUtils\PatchFinetune\ErrorFactor"
    filesets = []

    for file in os.listdir(error_factor_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(error_factor_dir, file)
            filesets.append(file_path)

    # 读取已有运行状态
    run_status_df = load_run_status(run_record_path)

    for stat_codes in stat_codes_df.iloc[0:3,]["stat_codes"]:

        stat_codes = str(stat_codes)

        # =====================================================
        # 1) 如果已经成功，直接跳过
        # =====================================================
        current_status = get_station_status(run_status_df, stat_codes)
        if current_status == "success":
            print(f"[跳过] 站点 {stat_codes} 已成功完成")
            continue

        print(f"\n================ 开始处理站点 {stat_codes} ================\n")

        try:
            # =================================================
            # 2) 匹配 TrainDist_path
            # =================================================
            TrainDist_path = None
            for file_path in filesets:
                file_name = os.path.basename(file_path)
                if stat_codes in file_name:
                    TrainDist_path = file_path
                    break

            if TrainDist_path is None:
                raise FileNotFoundError(f"没有找到与站点 {stat_codes} 对应的 ErrorFactor CSV")

            print(f"匹配到文件: {TrainDist_path}")

            # =================================================
            # 3) 检查站点信息
            # =================================================
            matched_stat = stat_df[stat_df["电站编号"] == stat_codes]
            if matched_stat.empty:
                raise ValueError(f"station_324_list.xlsx 中未找到站点 {stat_codes}")

            stat_type = matched_stat["电站二级"].values[0]
            max_num = matched_stat["站点模块数"].values[0]

            cat_name = f"{stat_codes}_{stat_type}.parquet"
            file_parquet = os.path.join(base_files, cat_name)

            print(f"站点类型: {stat_type}")
            print(f"模块数: {max_num}")
            print(f"Parquet路径: {file_parquet}")

            # =================================================
            # 4) 正式运行
            # =================================================
            result = ProjectModulesPlan(stat_codes, TrainDist_path)

            selected_df = result["selected_df"]

            if selected_df is not None and not selected_df.empty:
                selected_eps = ";".join([f"{x:.2f}" for x in selected_df["selected_eps"].tolist()])
                target_risks = ";".join([f"{x:.4f}" for x in selected_df["target_risk"].tolist()])
                actual_risks = ";".join([f"{x:.4f}" for x in selected_df["actual_risk"].tolist()])
            else:
                selected_eps = ""
                target_risks = ""
                actual_risks = ""

            record = {
                "stat_codes": stat_codes,
                "status": "success",
                "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary_path": result["summary_path"],
                "selected_map_path": result["selected_map_path"],
                "output_files": ";".join(result["saved_files"]),
                "selected_eps": selected_eps,
                "target_risks": target_risks,
                "actual_risks": actual_risks,
                "message": "运行成功"
            }

            run_status_df = upsert_station_status(run_status_df, record)
            save_run_status(run_status_df, run_record_path)

            print(f"[成功] 站点 {stat_codes} 已记录为 success")

        except Exception as e:
            err_msg = str(e)
            print(f"[失败] 站点 {stat_codes}: {err_msg}")

            record = {
                "stat_codes": stat_codes,
                "status": "failed",
                "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary_path": "",
                "selected_map_path": "",
                "output_files": "",
                "selected_eps": "",
                "target_risks": "",
                "actual_risks": "",
                "message": err_msg
            }

            run_status_df = upsert_station_status(run_status_df, record)
            save_run_status(run_status_df, run_record_path)

    print("Done.")