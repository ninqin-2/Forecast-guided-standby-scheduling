import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


def Train_model(model, train_loader, optimizer, criterion_asym_mae, criterion_nll, EPOCHS, DEVICE, test_loader, max_num, y_precise, save_file_path):
    print(f">>> Starting High-Performance Training on {DEVICE}...")
    best_mae = float("inf")
    best_state = None
    ## ====================================================================================
    ## ==================================训练循环===========================================
    ## ====================================================================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        # 注意：这里解包出 3 个变量
        for X, time_batch, y, y_indice in train_loader:
            X, time_batch, y, y_indice = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE), y_indice.to(DEVICE)
            last_observed_value = X[:, -1, 12:13]
            optimizer.zero_grad()

            # 前向传播 (传入时间特征)
            pred_params, kl_time = model(X, time_batch[:,y_precise,:], y_indice[:,y_precise,:])
            mu = pred_params[:, :, 0]

            # 混合 Loss
            loss_val = criterion_asym_mae(pred_params[:, :, 0], y[:,y_precise], time_batch[:, y_precise, 1])
            loss_dist = criterion_nll(pred_params, y[:,y_precise])
            loss = 1.0 * loss_dist + 3 * loss_val #+ 1e-3 * kl_time
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        trues_list, preds_list = [], []
        with torch.no_grad():
            for X, time_batch, y, y_indice in test_loader:
                X, time_batch, y, y_indice= X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE), y_indice.to(DEVICE)
                pred_params, _ = model(X, time_batch[:,y_precise,:], y_indice[:,y_precise,:])
                mu = pred_params[:, :, 0]

                preds_list.append(mu.cpu().numpy())
                trues_list.append(y[:,y_precise].cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)

        # 取整计算指标
        preds_int = np.rint(preds).astype(int)
        preds_int[preds_int > max_num] = max_num
        trues_int = np.rint(trues).astype(int)

        mae = mean_absolute_error(trues_int, preds_int)

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MAE: {mae:.4f} *Best*")
        else:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MAE: {mae:.4f}")

    from pathlib import Path

    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f">>> Saving best model to: {save_file_path}")
    print(f">>> Checkpoint folder exists: {save_file_path.parent.exists()}")

    # Use a Python file handle to avoid Windows/PyTorch Unicode-path issues.
    # 使用 Python 文件句柄保存，规避 Windows + PyTorch 对中文路径支持不稳定的问题。
    with open(save_file_path, "wb") as f:
        torch.save(best_state, f)
