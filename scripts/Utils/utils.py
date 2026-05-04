import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ================= 2. 数据集 (Dataset)  =================
class TSForecastDataset(Dataset):
    def __init__(self, X, y, time_X):
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

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 返回 5 个值: X, X_mark, y, Week_Y, Hour_Y
        return self.X[idx], self.time_X[idx], self.y[idx]

# --- 修改 DataLoader 构建函数 ---
def build_loaders(X_train, y_train, Time_train_emb,
                  X_test, y_test, Time_test_emb, batch_size=64):

    train_ds = TSForecastDataset(X_train, y_train, Time_train_emb)
    test_ds = TSForecastDataset(X_test, y_test, Time_test_emb)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader