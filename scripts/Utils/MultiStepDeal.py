import pandas as pd
import numpy as np

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

from scipy.stats import nbinom

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

from numpy.polynomial.hermite import hermgauss
def find_m_vectorized_gh_hour(
    r_array, p_array, hour_array,
    hour_param, global_mu, global_sigma,
    eps=0.05, m_max=400, K=25
):
    x, w = hermgauss(K)
    T = len(r_array)

    m_star = np.zeros(T, dtype=int)
    ft_star = np.zeros(T, dtype=float)
    target = 1.0 - eps

    for t in range(T):
        r = float(r_array[t])
        p = float(p_array[t])
        h = int(hour_array[t])

        mu_err, sigma_err, _ = hour_param.get(h, (global_mu, global_sigma, 0))
        sigma_err = max(float(sigma_err), 1e-6)

        for m in range(m_max + 1):
            ft = Ft_gh(m, r, p, mu_err, sigma_err, x, w)
            if ft >= target:
                m_star[t] = m
                ft_star[t] = ft
                break

    return m_star, ft_star