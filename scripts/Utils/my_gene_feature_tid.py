import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class mydefine():
    def __init__(self, path, TARGET_COL_NAME, SEQ_LENGTH, PRED_LENGTH=1, norm=False):
        self.path = path
        self.TARGET_COL_NAME = TARGET_COL_NAME
        self.scaler_type = "minmax"
        self.seq_length = SEQ_LENGTH
        self.pred_len = PRED_LENGTH

        df, feature_cols = self.load_and_process_data()
        self.data = df
        self.feature_cols = feature_cols
        self.norm=norm
        self.normalize_features()

    def read_parquet(self, path, n=5000, columns=None):
        df = pd.read_parquet(path)
        df_indexed = df.set_index('time')
        agg_rules = {
            # 取最大值 (Max)
            'num_modular': 'max', 'num_charging': 'max',

            # 取总和 (Sum) - 包含各种模型和SOC计数
            'start_model0_soc0': 'sum', 'start_model0_soc1': 'sum', 'start_model0_soc2': 'sum',
            'start_model0_soc3': 'sum', 'start_model0_soc4': 'sum', 'start_model1_soc0': 'sum',
            'start_model1_soc1': 'sum', 'start_model1_soc2': 'sum', 'start_model1_soc3': 'sum',
            'start_model1_soc4': 'sum', 'start_model2_soc0': 'sum', 'start_model2_soc1': 'sum',
            'start_model2_soc2': 'sum', 'start_model2_soc3': 'sum', 'start_model2_soc4': 'sum',
            'start_soc0': 'sum', 'start_soc1': 'sum', 'start_soc2': 'sum',
            'start_soc3': 'sum', 'start_soc4': 'sum',
            'model0': 'sum', 'model1': 'sum', 'model2': 'sum', 'num_coming': 'sum',

            # 取平均值 (Average)
            'Power_kW': 'mean','SFC_SW': 'mean',
            'T2M': 'mean', 'QV2M': 'mean', 'RH2M': 'mean',
            'PS': 'mean', 'WS10M': 'mean', 'electric_price': 'mean',

            # 取第一个值 (First) - 标签和静态信息
            'weekday': 'first', 'hour': 'first', 'minute': 'first',
            'day_type': 'first', 'day_type_label': 'first',
            'FID': 'first', 'FID_name': 'first'
        }

        # 3. 执行重采样 (以 30min 为例，'30min' 可替换为 '20min', '40min' 等)
        interval = '20min'
        resampled_df = df_indexed.resample(interval).agg(agg_rules)
        resampled_df = resampled_df.reset_index()
        if n is None:
            return resampled_df
        else:
            return resampled_df[:n]
    def load_and_process_data(self):
        print(">>> Loading and processing data...")
        df = self.read_parquet(self.path)
        df.columns = [c.strip() for c in df.columns]
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['min_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['min_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        contant_stat_info = ["num_modular", "num_charging", "num_coming"]
        generated_cols = []
        for c in contant_stat_info:
            col = f"L_{c}"
            df[col] = np.log1p(df[c])
            generated_cols.append(col)

        interactions = [("num_coming", "num_modular"), ("num_coming", "num_charging"),
            ("L_num_coming", "L_num_modular"), ("L_num_coming", "L_num_charging")]

        for a, b in interactions:
            col = f"{a}_x_{b}"
            df[col] = df[a] * df[b]
            generated_cols.append(col)

        df['target_raw'] = df[self.TARGET_COL_NAME].fillna(0).astype(int)
        df['target_binary'] = (df['target_raw'] > 0).astype(float)
        add_cols = [f"MIN_{i+1}" for i in range(20)]
        base_cols = ["weekday", "hour", "minute", "hour_sin", "hour_cos", "min_sin", "min_cos", "electric_price",
                     "max_stop", "SFC_SW", "T2M", "QV2M", "RH2M", "PS", "WS10M","total_energy_kwh"] + contant_stat_info
        decrete_cols = [f"start_model{i}_soc{j}" for i in [0, 1, 2] for j in [0, 1, 2, 3, 4]] + \
                       [f"start_soc{i}" for i in [0, 1, 2, 3, 4]] + \
                       [f"model{i}" for i in [0, 1, 2]]
        all_features = generated_cols + base_cols + decrete_cols

        return df, all_features



    def normalize_features(self):
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        valid_columns = [col for col in self.feature_cols if col in self.data.columns]
        X = self.data[valid_columns].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)  # 替换正负无穷为 NaN
        X = X.fillna(0.0)
        if self.norm:
            # fit_transform
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        self.X_scaled = X_scaled

    def create_stratified_sequences(self, train_ratio=0.8, slots_per_day=72, step_minutes=20):
        X_data = self.X_scaled
        y_data = self.data["target_raw"].values
        seq_len = self.seq_length
        pred_len = self.pred_len

        # ===== ✅ 只用 start_hour / start_minute 生成 tid，最稳 =====
        # 强制转成数值（字符串/空值都会变 NaN）
        hour = pd.to_numeric(self.data["hour"], errors="coerce").fillna(0).astype(int).values
        minute = pd.to_numeric(self.data["minute"], errors="coerce").fillna(0).astype(int).values

        # 清洗：hour限制在[0,23]，minute限制在[0,59]
        hour = np.clip(hour, 0, 23)
        minute = np.clip(minute, 0, 59)

        minutes_of_day = hour * 60 + minute  # 0..1439
        tid_data = (minutes_of_day // step_minutes).astype(int)  # 0..71 (20min)
        tid_data = np.clip(tid_data, 0, slots_per_day - 1)

        # ===== 生成序列 =====
        X_train_list, y_train_list, week_train_list, hour_train_list, tid_train_list = [], [], [], [], []
        X_test_list, y_test_list, week_test_list, hour_test_list, tid_test_list = [], [], [], [], []

        indices = self.data.index
        curr_X, curr_y, curr_tid, curr_h, curr_w = [], [], [], [], []
        for i in range(0, len(indices) - seq_len - pred_len + 1):
            curr_X.append(X_data[indices[i:i + seq_len]])

            y_idx = indices[i + seq_len: i+seq_len+pred_len]  # 目标点在全局data中的索引
            curr_y.append(y_data[y_idx])
            curr_w.append(X_data[y_idx, 12])
            curr_h.append(X_data[y_idx, 13])

        n_train = int(len(curr_X) * train_ratio)

        X_train_list.extend(curr_X[:n_train])
        y_train_list.extend(curr_y[:n_train])
        week_train_list.extend(curr_w[:n_train])
        hour_train_list.extend(curr_h[:n_train])
        tid_train_list.extend(curr_tid[:n_train])

        X_test_list.extend(curr_X[n_train:])
        y_test_list.extend(curr_y[n_train:])
        week_test_list.extend(curr_w[n_train:])
        hour_test_list.extend(curr_h[n_train:])
        tid_test_list.extend(curr_tid[n_train:])

        return (np.array(X_train_list), np.array(y_train_list), np.array(tid_train_list), np.array(week_train_list), np.array(hour_train_list),
                np.array(X_test_list), np.array(y_test_list), np.array(tid_test_list), np.array(week_test_list), np.array(hour_test_list) )


if __name__ == "__main__":
    # ==========================================
    # 1. 全局配置
    # ==========================================
    FILE_CANDIDATES = [
        os.path.join("Data_gene_load_file", "charge_analysis_results_interval20.csv"),
        r"DATA\上海交运上南路3459号充电站.csv",
        r"/home/tjluser/CODE_libs/NQ/Download/Data_gene_load_file/charge_analysis_results_interval20.csv",
        os.path.join(r"E:\CODE_libs2025\PredictModulesUtils\DataSource\产业园区_150_25W次\3101140854_标准公共快充站.parquet")]

    FILE_PATH = next(p for p in FILE_CANDIDATES if os.path.exists(p))
    TARGET_COL_NAME = 'model2'
    SEQ_LENGTH=72
    next_data = mydefine(FILE_PATH, TARGET_COL_NAME, SEQ_LENGTH)

    X_train, y_train, tid_train, week_train, hour_train, X_test, y_test, tid_test, week_test, hour_test = next_data.create_stratified_sequences()
