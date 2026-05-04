import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_uncertainty_correlation(sigma_hourly, alpha_mean,Figure_path):
    """
    计算 Sigma 和 Alpha 之间的统计相关性
    """
    # 转换为 numpy 数组
    s = np.array(sigma_hourly)
    a = np.array(alpha_mean)

    # 计算相关系数
    p_corr, p_pvalue = pearsonr(s, a)
    s_corr, s_pvalue = spearmanr(s, a)

    print(">>> Uncertainty Correlation Analysis <<<")
    print(f"Pearson Correlation (线性相关性): {p_corr:.4f}, p-value: {p_pvalue:.4f}")
    print(f"Spearman Correlation (等级相关性): {s_corr:.4f}, p-value: {s_pvalue:.4f}")

    # 散点回归图可视化
    plt.figure(figsize=(8, 6))
    sns.regplot(x=s, y=a, ci=95,
                scatter_kws={'s': 50, 'color': 'tab:blue'},
                line_kws={'color': 'tab:red', 'label': f'Pearson R={p_corr:.2f}'})

    plt.title("Correlation: Embedding Sigma vs. Output Alpha")
    plt.xlabel("Embedding Sigma (Random Effect)")
    plt.ylabel("Output Alpha (Uncertainty)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(rf"{Figure_path}/Uncertainty Correlation Analysis.png", dpi=300, bbox_inches='tight')



def monitor_uncertainty_balance(model, test_loader, DEVICE, y_precise, Figure_path):
    """
    对比 24 小时的嵌入层随机强度 (Sigma) 和 输出层不确定性 (Alpha)
    """
    model.eval()
    hourly_alpha = [[] for _ in range(24)]

    # 1. 采集测试集中的 Alpha 表现
    with torch.no_grad():
        for X, time_batch, y, y_indice in test_loader:
            X, time_batch, y, y_indice = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE), y_indice.to(DEVICE)
            pred_params, _ = model(X, time_batch[:, y_precise, :], y_indice[:,y_precise,:])

            # alpha 位于 pred_params 的最后一个维度
            alpha = pred_params[:, :, 1].cpu()  # [B, pred_len]
            hours = time_batch[:, y_precise, 1].cpu().long()  # 取预测起始时刻的小时

            for i in range(X.size(0)):
                h = hours[i].item() % 24
                hourly_alpha[h].append(alpha[i].mean().item())

    # 计算 Alpha 的 24 小时均值
    alpha_mean = [np.mean(val) if val else 0 for val in hourly_alpha]

    # 2. 提取 Sigma (从 TimeRandomEffectEmbedding)
    logstd_weight = model.time_embedding.hour_logstd.weight.detach().cpu()
    sigma_hourly = torch.exp(logstd_weight).mean(dim=-1).numpy()

    # 3. 双子图绘制
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 绘制 Sigma (柱状图)
    color_sigma = 'tab:blue'
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Embedding Sigma (Input Randomness)', color=color_sigma, fontsize=12)
    ax1.bar(range(24), sigma_hourly, color=color_sigma, alpha=0.3, label='Sigma (Embedding)')
    ax1.tick_params(axis='y', labelcolor=color_sigma)
    ax1.set_ylim(0, max(sigma_hourly) * 1.5)

    # 绘制 Alpha (折线图)
    ax2 = ax1.twinx()
    color_alpha = 'tab:red'
    ax2.set_ylabel('Predicted Alpha (Output Uncertainty)', color=color_alpha, fontsize=12)
    ax2.plot(range(24), alpha_mean, color=color_alpha, marker='o', linewidth=2, label='Alpha (Output)')
    ax2.tick_params(axis='y', labelcolor=color_alpha)

    plt.title("Uncertainty Decomposition: Embedding Sigma vs. Output Alpha", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig(rf"{Figure_path}/Uncertainty Decomposition.png", dpi=300, bbox_inches='tight')

    # 调用分析
    # sigma_hourly 和 alpha_mean 是你上一步监控函数算出的结果
    analyze_uncertainty_correlation(sigma_hourly, alpha_mean, Figure_path)

def analyze_temporal_bias_comparison(model, train_loader, test_loader, DEVICE, y_precise, Figure_path):
    """
    对比分析训练集和测试集在不同时间点(Hour/Weekday)的 MAE。
    依赖 Dataset 返回的真实标签时间 (Week_Y, Hour_Y)。
    """
    print("\n>>> Starting Comparative Temporal Bias Analysis (Train vs Test)...")
    model.eval()

    # 定义辅助函数：处理单个 Loader
    def process_loader(loader, name):
        all_mae = []
        all_hours = []
        all_weeks = []
        all_residuals = []

        with torch.no_grad():
            for X, time_batch, y, y_indice in test_loader:
                X, time_batch, y, y_indice = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE), y_indice.to(DEVICE)

                # 预测
                pred_params, _ = model(X, time_batch[:, y_precise, :], y_indice[:,y_precise,:])
                mu = pred_params[:, :, 0]  # [B, Pred_Len]

                # 计算绝对误差
                abs_error = torch.abs(y[:, y_precise] - mu).cpu().numpy().flatten()
                residual = (y[:, y_precise] - mu).cpu().numpy().flatten()

                # 提取对应的真实时间和星期
                # w_y, h_y shape [B, Pred_Len] -> flatten
                h_y = time_batch[:,y_precise,1].cpu().numpy().flatten()
                w_y = time_batch[:,y_precise,0].cpu().numpy().flatten()

                all_mae.append(abs_error)
                all_residuals.append(residual)
                all_hours.append(h_y)
                all_weeks.append(w_y)

        return pd.DataFrame({
            'Hour': np.concatenate(all_hours),
            'Weekday': np.concatenate(all_weeks),
            'MAE': np.concatenate(all_mae),
            'Residual': np.concatenate(all_residuals),
            'Dataset': name
        })

    # 1. 分别处理数据
    print("    Processing Train Set...")
    df_train = process_loader(train_loader, "Train")
    print("    Processing Test Set...")
    df_test = process_loader(test_loader, "Test")

    # 2. 准备绘图
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    # 创建画布
    fig = plt.figure(figsize=(20, 12))
    plt.suptitle("Temporal Bias Comparison: Train vs Test", fontsize=16)

    # --- 子图 1 & 2: MAE 热力图对比 ---
    # 统一色阶范围，方便对比颜色深浅
    max_mae = max(df_train['MAE'].mean() * 2, df_test['MAE'].mean() * 2)

    for idx, (df, title) in enumerate([(df_train, "Train Set MAE"), (df_test, "Test Set MAE")]):
        ax = plt.subplot(2, 2, idx + 1)

        # 透视表: Weekday x Hour -> Mean MAE
        pivot = df.pivot_table(index='Weekday', columns='Hour', values='MAE', aggfunc='mean')
        pivot = pivot.reindex(index=range(7), columns=range(24))  # 补全

        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f",
                    vmin=0, vmax=max_mae, cbar=True, ax=ax)

        ax.set_yticklabels([weekday_map[i] for i in range(7)], rotation=0)
        ax.set_title(f"{title} (Heatmap)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")

    # --- 子图 3 & 4: 残差箱线图对比 (按小时) ---
    # 统一 Y 轴范围
    y_lim = np.percentile(np.concatenate([df_train['Residual'], df_test['Residual']]), [1, 99])

    for idx, (df, title) in enumerate([(df_train, "Train Residuals"), (df_test, "Test Residuals")]):
        ax = plt.subplot(2, 2, idx + 3)

        sns.boxplot(x='Hour', y='Residual', data=df, showfliers=False, palette="coolwarm", ax=ax)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        ax.set_ylim(y_lim)  # 统一刻度
        ax.set_title(f"{title} by Hour (True - Pred)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig(rf"{Figure_path}/Residual by Hour.png", dpi=300, bbox_inches='tight')



def monitor_hourly_stochasticity(model, Figure_path):
    """
    提取并可视化模型学到的 24 小时随机效应强度 (Sigma)
    """
    model.eval()
    with torch.no_grad():
        # 1. 从 TimeRandomEffectEmbedding 中提取 hour_logstd 的权重
        # 路径: model -> time_embedding -> hour_logstd
        logstd_weight = model.time_embedding.hour_logstd.weight.detach().cpu()

        # 2. 计算实际的标准差 sigma = exp(log_std)
        # 对 d_model 维度取平均，代表该小时总体的随机强度
        sigma_hourly = torch.exp(logstd_weight).mean(dim=-1).numpy()

        # 3. 绘图
        plt.figure(figsize=(12, 5))
        sns.barplot(x=list(range(24)), y=sigma_hourly, palette="viridis")

        plt.title("Hourly Random Effect Intensity (Sigma)", fontsize=14)
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Mean Learned Sigma", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 标注最大随机时刻
        max_hour = sigma_hourly.argmax()
        plt.annotate(f'Peak Uncertainty: {max_hour}:00',
                     xy=(max_hour, sigma_hourly[max_hour]),
                     xytext=(max_hour + 1, sigma_hourly[max_hour] * 1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        plt.savefig(rf"{Figure_path}/Hourly Random Effect Intensity (Sigma).png", dpi=300, bbox_inches='tight')


def Predictarange(preds, alphas, limit, trues_int, preds_int, max_num, mae, Figure_path):
    # 计算置信区间 (Negative Binomial Variance = mu + mu^2/alpha)
    variance = preds + (preds ** 2) / (alphas + 1e-6)
    std_dev = np.sqrt(variance)

    # 画图
    plt.figure(figsize=(15, 6))
    x_range = np.arange(limit)

    plt.plot(trues_int[:limit, 0], label='Ground Truth', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(preds_int[:limit, 0], label='Pred Mean', color='#007acc', linestyle='--', linewidth=1.5)

    # 90% 置信区间 (Mean +/- 1.96 * Std)
    lower_bound = np.maximum(0, preds - 1.28 * std_dev)  # 负荷不能小于0
    upper_bound = preds + 1.28 * std_dev
    upper_bound = np.clip(upper_bound, None, max_num)  # None 表示没有下限，48 是上限
    plt.fill_between(x_range,
                     lower_bound[:limit, 0],
                     upper_bound[:limit, 0],
                     color='#007acc', alpha=0.15, label='80% Confidence Interval')

    plt.title(f"High-Performance Probabilistic Forecast (MAE={mae:.3f})")
    plt.xlabel("Sample Index")
    plt.ylabel("Charging Load (Modules)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf"{Figure_path}/High-Performance Probabilistic Forecast.png", dpi=300, bbox_inches='tight')





def Pred_model(next_data, model, train_loader, test_loader, DEVICE, CSV_SAVE_DIR, limit, max_num, y_precise, save_file_path, stat_codes):
    Figure_path = rf"Figures/{stat_codes}"
    if not os.path.exists(Figure_path):
        os.makedirs(Figure_path)

    best_state = torch.load(save_file_path)
    # ================= 7. 最终评估与可视化 =================
    print("\n>>> Final Evaluation (Best Model) <<<")
    model.load_state_dict(best_state)
    model.eval()

    mus_list, alphas_list, trues_list, weeks_list, hours_list = [], [], [], [], []

    with torch.no_grad():
        for X, time_batch, y, y_indice in test_loader:
            X, time_batch, y, y_indice = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE), y_indice.to(DEVICE)
            pred_params,_ = model(X, time_batch[:,y_precise,:], y_indice[:,y_precise,:])

            mus_list.append(pred_params[:, :, 0].cpu().numpy())
            alphas_list.append(pred_params[:, :, 1].cpu().numpy())
            trues_list.append(y[:,y_precise].cpu().numpy())
            weeks_list.append(time_batch[:,y_precise,0].cpu().numpy())  # 添加 batch_weeks 到结果列表
            hours_list.append(time_batch[:,y_precise,1].cpu().numpy())  # 添加 batch_hours 到结果列表

    preds = np.concatenate(mus_list, axis=0)
    alphas = np.concatenate(alphas_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    weeks = np.concatenate(weeks_list, axis=0)  # 合并 week 数据
    hours = np.concatenate(hours_list, axis=0)  # 合并 hour 数据

    preds_int = np.rint(preds).astype(int)
    preds_int[preds_int > max_num] = max_num
    trues_int = np.rint(trues).astype(int)

    mae = mean_absolute_error(trues_int, preds_int)
    rmse = mean_squared_error(trues_int, preds_int, squared=False)
    r2 = r2_score(trues_int, preds_int, multioutput="uniform_average")

    print(f"Final MAE : {mae:.4f}")
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final R2  : {r2:.4f}")



    # 保存结果到 CSV 文件
    pd.DataFrame({
        'Time': next_data.test_time[:, 0],
        'True': trues_int[:, 0],
        'Pred_Mu': preds_int[:, 0],
        'Pred_Alpha': alphas[:, 0],
        'Error': trues_int[:, 0] - preds_int[:, 0],
        'Week': weeks[:,0],  # 添加 Week 列
        'Hour': hours[:,0]   # 添加 Hour 列
    }).to_csv(CSV_SAVE_DIR, index=False)
    print(f"Results saved to {CSV_SAVE_DIR}")

    err =  trues_int[:, 0] - preds_int[:, 0]
    err = err[~np.isnan(err)]
    pos_err = err[err > 0]
    q90 = np.quantile(pos_err, 0.90)
    q95 = np.quantile(pos_err, 0.95)
    q99 = np.quantile(pos_err, 0.99)
    max_val = np.max(pos_err)
    print("预测区段最坏情况：")
    print(f"q90 : {q90:.6f}")
    print(f"q95 : {q95:.6f}")
    print(f"q99 : {q99:.6f}")
    print(f"max : {max_val:.6f}")

    # ================= 调用 =================
    if 'model' in locals():
        Predictarange(preds, alphas, limit, trues_int, preds_int, max_num, mae, Figure_path)
        analyze_temporal_bias_comparison(model, train_loader, test_loader, DEVICE, y_precise, Figure_path)
        monitor_hourly_stochasticity(model, Figure_path)
        monitor_uncertainty_balance(model, test_loader, DEVICE, y_precise, Figure_path)

    return mae, rmse, r2