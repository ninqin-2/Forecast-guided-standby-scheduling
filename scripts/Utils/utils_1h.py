import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ================= 2. 数据集 (Dataset)  =================
class TSForecastDataset(Dataset):
    def __init__(self, X, y, time_X, yindices):
        """
        X:       [N, Seq_Len, C]
        y:       [N, Pred_Len]
        tids:    [N, Seq_Len] (用于生成输入的 x_mark)
        weeks_y: [N, Pred_Len] (标签对应的真实星期) <--- 新增
        hours_y: [N, Pred_Len] (标签对应的真实小时) <--- 新增
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.time_X = torch.from_numpy(time_X.astype(np.float32))
        self.yindices = torch.from_numpy(yindices.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 返回 5 个值: X, X_mark, y, Week_Y, Hour_Y
        return self.X[idx], self.time_X[idx], self.y[idx], self.yindices[idx]

# --- 修改 DataLoader 构建函数 ---
def build_loaders(X_train, y_train, Time_train_emb, y_train_elePrice, X_test, y_test, Time_test_emb, y_test_elePrice, batch_size=64):

    train_ds = TSForecastDataset(X_train, y_train, Time_train_emb, y_train_elePrice)
    test_ds = TSForecastDataset(X_test, y_test, Time_test_emb, y_test_elePrice)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.stats import nbinom
from matplotlib.ticker import MultipleLocator

def PlotModulesOpenPlan(eps, max_num, xlim_set, out_df, savepath=None):
    mpl.rcParams.update({"font.family": "SimHei",  "font.size": 14, "axes.unicode_minus": False,
        "lines.antialiased": True})
    Heps = str(1-eps)
    bg_color = "#FCFCFC"
    color_true = "#404272"  # 深海蓝 (True)
    color_q = "#A51C30"     # 深砖红 (Quantile)
    color_qq = "#2baeb3"
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    x = np.arange(len(out_df))
    true_values = out_df["true_0"].values
    q_values = out_df["m_final"].values
    line_q, = ax.plot(x, q_values,
                      color=color_q, linestyle=(0, (5, 2)), linewidth=1.8,
                      label=f"预测上界 (Q$_{Heps}$ + Error$_{Heps}$)", zorder=3)
    ax.fill_between(x, 0, true_values,
                    color=color_true, alpha=0.12,
                    label="置信区间 (90% Prob + 90%Error)", zorder=1)
    line_true, = ax.plot(x, true_values,
                        color=color_true, linewidth=1.2,
                        label="观测真实值", zorder=4)
    ax.set_ylabel("数值", labelpad=10)
    ax.set_xlim(xlim_set[0], xlim_set[1])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.legend(
        handles=[line_true, line_q],
        loc="upper right",
        frameon=True,
        facecolor=bg_color,
        edgecolor="none",
        fontsize=12,
        framealpha=1
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(str(savepath), bbox_inches='tight', dpi=300)
        print(f"📊 图像已保存至: {savepath}")
    else:
        plt.show()