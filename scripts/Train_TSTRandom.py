import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def Train_model(model, train_loader, optimizer, criterion_asym_mae, criterion_nll, EPOCHS, DEVICE, test_loader, max_num, save_file_path):
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
        for X, time_batch, y in train_loader:
            X, time_batch, y = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播 (传入时间特征)
            pred_params, kl_time = model(X, time_batch)
            mu = pred_params[:, :, 0]

            # 混合 Loss
            loss_val = criterion_asym_mae(pred_params[:, :, 0], y, time_batch[:, :, 1])
            loss_dist = criterion_nll(pred_params, y)
            loss = 1.0 * loss_dist + 3 * loss_val + 1e-3 * kl_time
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        trues_list, preds_list = [], []
        with torch.no_grad():
            for X, time_batch, y in test_loader:
                X, time_batch, y = X.to(DEVICE), time_batch.to(DEVICE), y.to(DEVICE)
                pred_params, _ = model(X, time_batch)
                mu = pred_params[:, :, 0]

                preds_list.append(mu.cpu().numpy())
                trues_list.append(y.cpu().numpy())

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


    torch.save(best_state, save_file_path)

