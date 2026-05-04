import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')


class mydefine():
    def __init__(self, path, TARGET_COL_NAME, SEQ_LENGTH, PRED_LENGTH=1, norm=False):
        self.path = path
        self.TARGET_COL_NAME = TARGET_COL_NAME
        self.scaler_type = "minmax"
        self.seq_length = SEQ_LENGTH
        self.pred_len = PRED_LENGTH

        df, feature_cols, Time_cols = self.load_and_process_data()
        self.data = df
        self.feature_cols = feature_cols
        self.Time_cols = Time_cols
        self.norm=norm
        self.normalize_features()

    def read_parquet(self, path, n=None, columns=None):
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

    def read_parquet_compact(self, path):

        df = self.read_parquet(path)

        # float64 -> float32
        float64_cols = df.select_dtypes(include="float64").columns
        df[float64_cols] = df[float64_cols].astype("float32")

        # int64 / int32 -> int16（安全检查）
        int_cols = df.select_dtypes(include=["int32", "int64"]).columns
        for col in int_cols:
            if df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype("int16")

        return df

    def load_and_process_data(self):
        print(">>> Loading and processing data...")
        df = self.read_parquet_compact(self.path)
        df.columns = [c.strip() for c in df.columns]
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype("float32")
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype("float32")
        df['min_sin'] = np.sin(2 * np.pi * df['minute'] / 60).astype("float32")
        df['min_cos'] = np.cos(2 * np.pi * df['minute'] / 60).astype("float32")
        contant_stat_info = ["num_modular", "num_charging", "num_coming"]
        """generated_cols = []
        for c in contant_stat_info:
            col = f"L_{c}"
            df[col] = np.log1p(df[c])
            generated_cols.append(col)

        interactions = [
            ("num_coming", "ratio_using"), ("num_coming", "num_modular"),
            ("num_coming", "num_charging"), ("ratio_using", "num_modular"), ("L_num_coming", "ratio_using"),
            ("L_num_coming", "L_num_modular"), ("L_num_coming", "L_num_charging"), ("ratio_using", "L_num_modular")]

        for a, b in interactions:
            col = f"{a}_x_{b}"
            df[col] = df[a] * df[b]
            generated_cols.append(col)

        df['target_raw'] = df[self.TARGET_COL_NAME].fillna(0).astype(int)
        df['target_binary'] = (df['target_raw'] > 0).astype(float)"""

        base_cols = ["hour_sin", "hour_cos", "min_sin", "min_cos", "electric_price",
                     "Power_kW", "SFC_SW", "T2M", "QV2M", "RH2M", "PS", "WS10M"] + contant_stat_info
        decrete_cols = [f"start_model{i}_soc{j}" for i in [0, 1, 2] for j in [0, 1, 2, 3, 4]] + \
                       [f"start_soc{i}" for i in [0, 1, 2, 3, 4]] + \
                       [f"model{i}" for i in [0, 1, 2]]

        Time_cols = ["weekday", "hour", "minute", "day_type"]
        all_features = base_cols + decrete_cols + Time_cols

        return df, all_features, Time_cols



    def normalize_features(self):
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        valid_fea_columns = [col for col in self.feature_cols if col in self.data.columns]
        X1 = self.data[valid_fea_columns].apply(pd.to_numeric, errors="coerce")
        X1 = X1.replace([np.inf, -np.inf], np.nan)  # 替换正负无穷为 NaN
        X1 = X1.fillna(0.0)

        valid_time_columns = [col for col in self.Time_cols if col in self.data.columns]
        X2 = self.data[valid_time_columns].apply(pd.to_numeric, errors="coerce")
        X2 = X2.replace([np.inf, -np.inf], np.nan)  # 替换正负无穷为 NaN
        X2 = X2.fillna(0.0)

        if self.norm:
            # fit_transform
            X_scaled = self.scaler.fit_transform(X1)
        else:
            fea_scaled = X1.values
            time_scaled = X2.values
        self.fea_scaled = fea_scaled
        self.time_scaled = time_scaled

    def create_stratified_sequences(self, train_ratio=0.8, slots_per_day=72, step_minutes=20):
        X_data = self.fea_scaled
        T_data = self.time_scaled
        y_data = self.data[self.TARGET_COL_NAME].values
        seq_len = self.seq_length
        pred_len = self.pred_len

        #hour = pd.to_numeric(self.data["hour"], errors="coerce").fillna(0).astype(int).values
        #minute = pd.to_numeric(self.data["minute"], errors="coerce").fillna(0).astype(int).values
        #hour = np.clip(hour, 0, 23)
        #minute = np.clip(minute, 0, 59)
        #minutes_of_day = hour * 60 + minute  # 0..1439
        #tid_data = (minutes_of_day // step_minutes).astype(int)  # 0..71 (20min)
        #tid_data = np.clip(tid_data, 0, slots_per_day - 1)

        # ===== 生成序列 =====
        X_train_list, y_train_list, Time_train_emb = [], [], []
        X_test_list, y_test_list, Time_test_emb = [], [], []

        indices = self.data.index
        curr_X, curr_y, curr_T = [], [], []
        for i in range(0, len(indices) - seq_len - pred_len + 1):
            curr_X.append(X_data[indices[i:i + seq_len]])
            curr_T.append(T_data[indices[i:i + seq_len]])

            y_idx = indices[i + seq_len: i+seq_len+pred_len]  # 目标点在全局data中的索引
            curr_y.append(y_data[y_idx])


        n_train = int(len(curr_X) * train_ratio)

        X_train_list.extend(curr_X[:n_train])
        y_train_list.extend(curr_y[:n_train])
        Time_train_emb.extend(curr_T[:n_train])

        X_test_list.extend(curr_X[n_train:])
        y_test_list.extend(curr_y[n_train:])
        Time_test_emb.extend(curr_T[n_train:])

        return (np.array(X_train_list), np.array(y_train_list), np.array(Time_train_emb),
                np.array(X_test_list), np.array(y_test_list), np.array(Time_test_emb))


if __name__ == "__main__":
    # ==========================================
    # 1. 全局配置
    # ==========================================
    FILE_CANDIDATES = [
        os.path.join(r"E:\CODE_libs2025\PredictModulesUtils\DataSource\产业园区_150_25W次\3101140854_标准公共快充站.parquet")]

    FILE_PATH = next(p for p in FILE_CANDIDATES if os.path.exists(p))
    TARGET_COL_NAME = 'num_modular'
    SEQ_LENGTH=72
    next_data = mydefine(FILE_PATH, TARGET_COL_NAME, SEQ_LENGTH)

    X_train, y_train, Time_train_emb, X_test, y_test, Time_test_emb = next_data.create_stratified_sequences()
    print(Time_train_emb)
