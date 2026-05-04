import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os



def Pred_model(next_data, model, train_loader, test_loader, DEVICE, CSV_SAVE_DIR, max_num, y_precise, save_file_path, stat_codes, test_save=True):
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

    if test_save:

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

    else:
        return   pd.DataFrame({
            'True': trues_int[:, 0],
            'Pred_Mu': preds_int[:, 0],
            'Pred_Alpha': alphas[:, 0],
            'Error': trues_int[:, 0] - preds_int[:, 0],
            'Week': weeks[:,0],
            'Hour': hours[:,0]})
