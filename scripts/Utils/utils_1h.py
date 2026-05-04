import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


# =============================================================================
# Dataset and dataloader
# =============================================================================

class TSForecastDataset(Dataset):
    """
    Dataset for time-series module-demand forecasting.

    用于模块需求时间序列预测的数据集。
    """

    def __init__(self, X, y, time_X, yindices):
        """
        Parameters
        ----------
        X : ndarray
            Input sequence, shape [N, seq_len, C].
        y : ndarray
            Prediction target, shape [N, pred_len].
        time_X : ndarray
            Time features of the prediction horizon, shape [N, pred_len, C_time].
        yindices : ndarray
            Future-known exogenous features of the prediction horizon.
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.time_X = torch.from_numpy(time_X.astype(np.float32))
        self.yindices = torch.from_numpy(yindices.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.time_X[idx], self.y[idx], self.yindices[idx]


def build_loaders(
    X_train,
    y_train,
    Time_train_emb,
    y_train_elePrice,
    X_test,
    y_test,
    Time_test_emb,
    y_test_elePrice,
    batch_size=64,
):
    """
    Build train and test dataloaders.
    构建训练集与测试集 DataLoader。
    """
    train_ds = TSForecastDataset(
        X_train,
        y_train,
        Time_train_emb,
        y_train_elePrice,
    )

    test_ds = TSForecastDataset(
        X_test,
        y_test,
        Time_test_emb,
        y_test_elePrice,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


# =============================================================================
# Plotting
# =============================================================================

def set_plot_style() -> None:
    """
    Set English plotting style.
    设置英文绘图风格。
    """
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 13,
        "axes.unicode_minus": False,
        "lines.antialiased": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def _to_scalar(value) -> float:
    """
    Convert scalar-like input to float.
    将数组或标量转换为 float。
    """
    arr = np.asarray(value).reshape(-1)
    return float(arr[0])


def PlotModulesOpenPlan(eps, max_num, xlim_set, out_df, savepath=None):
    """
    Plot observed module demand and adjusted activation bound.

    绘制真实模块需求与修正后的启用模块上界。
    """
    set_plot_style()

    confidence_level = 1 - eps
    bg_color = "#FCFCFC"
    color_observed = "#404272"
    color_bound = "#A51C30"

    fig, ax = plt.subplots(
        figsize=(12, 4),
        dpi=300,
        facecolor=bg_color,
    )

    ax.set_facecolor(bg_color)

    x = np.arange(len(out_df))
    observed = out_df["true_0"].values
    adjusted_bound = out_df["m_final"].values

    line_bound, = ax.plot(
        x,
        adjusted_bound,
        color=color_bound,
        linestyle=(0, (5, 2)),
        linewidth=1.8,
        label=f"Adjusted activation bound (Q$_{{{confidence_level:.2f}}}$ + error buffer)",
        zorder=3,
    )

    ax.fill_between(
        x,
        0,
        observed,
        color=color_observed,
        alpha=0.12,
        label="Observed demand area",
        zorder=1,
    )

    line_observed, = ax.plot(
        x,
        observed,
        color=color_observed,
        linewidth=1.2,
        label="Observed module demand",
        zorder=4,
    )

    installed_modules = _to_scalar(max_num)

    ax.axhline(
        installed_modules,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="Installed modules",
        zorder=2,
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Number of modules")
    ax.set_title("Forecast-guided module activation bound")

    if xlim_set is not None:
        ax.set_xlim(xlim_set[0], xlim_set[1])

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(20))

    ax.legend(
        handles=[line_observed, line_bound],
        loc="upper right",
        frameon=True,
        facecolor=bg_color,
        edgecolor="none",
        fontsize=11,
        framealpha=1,
    )

    fig.tight_layout()

    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {savepath}")
        plt.close(fig)
    else:
        plt.show()