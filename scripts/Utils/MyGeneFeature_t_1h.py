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
        """
        Read already-aggregated 20-minute station-level data.

        读取已经完成 20 min 聚合的数据。

        Notes
        -----
        The input sample data are already resampled to 20-minute resolution.
        Therefore, this function does NOT resample again.

        输入数据已经是 20 min 粒度，因此这里不再重新 resample。
        """

        # ============================================================
        # 1. Read file
        # 1. 读取文件
        # ============================================================
        path_str = str(path)

        if path_str.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path_str.endswith(".csv"):
            df = pd.read_csv(path)
        elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        df.columns = [str(c).strip() for c in df.columns]

        # ============================================================
        # 2. Optional column subset
        # 2. 可选列筛选
        # ============================================================
        if columns is not None:
            keep_cols = [c for c in columns if c in df.columns]
            df = df[keep_cols].copy()

        # ============================================================
        # 3. Time column
        # 3. 时间列
        # ============================================================
        if "time" not in df.columns:
            raise KeyError(
                "Column 'time' is required for sequence generation. "
                "Please keep `time` in the temporary training file."
            )

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).copy()
        df = df.sort_values("time").reset_index(drop=True)

        # ============================================================
        # 4. Minimal schema compatibility
        # 4. 最小字段兼容
        # ============================================================
        # 只在旧字段不存在时，才从英文公开字段补一份旧字段。
        # 如果旧字段已经存在，不覆盖、不追加、不重排。
        compatibility_map = {
            "module_demand": "num_modular",
            "active_vehicle_count": "num_charging",
            "arriving_vehicle_count": "num_coming",
            "aggregate_power_kw": "Power_kW",
            "electricity_price": "electric_price",
            "raw_hour": "hour",
            "raw_minute": "minute",
            "station_id_anonymized": "FID",
            "station_name_anonymized": "FID_name",
        }

        for english_col, legacy_col in compatibility_map.items():
            if legacy_col not in df.columns and english_col in df.columns:
                df[legacy_col] = df[english_col]

        # 如果没有 day_type_label，但有 day_type，则补一列
        if "day_type_label" not in df.columns and "day_type" in df.columns:
            df["day_type_label"] = df["day_type"]

        # ============================================================
        # 5. Required legacy columns check
        # 5. 检查模型流程需要的旧字段
        # ============================================================
        required_cols = [
            "time",
            "num_modular",
            "num_charging",
            "num_coming",
            "Power_kW",
            "electric_price",
            "weekday",
            "hour",
            "minute",
            "day_type",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            raise KeyError(
                "Missing required columns for model input: "
                f"{missing_cols}"
            )

        if n is None:
            return df
        else:
            return df.iloc[:n].copy()

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

        base_cols = ["hour_sin", "hour_cos", "min_sin", "min_cos", "electric_price",
                     "Power_kW", "SFC_SW", "T2M", "QV2M", "RH2M", "PS", "WS10M"] + contant_stat_info
        decrete_cols = [f"start_model{i}_soc{j}" for i in [0, 1, 2] for j in [0, 1, 2, 3, 4]] + \
                       [f"start_soc{i}" for i in [0, 1, 2, 3, 4]] + \
                       [f"model{i}" for i in [0, 1, 2]]

        Time_cols = ["weekday", "hour", "minute", "day_type"]
        all_features =  base_cols + decrete_cols + Time_cols + generated_cols

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

        time_series = self.data["time"].values
        #hour = pd.to_numeric(self.data["hour"], errors="coerce").fillna(0).astype(int).values
        #minute = pd.to_numeric(self.data["minute"], errors="coerce").fillna(0).astype(int).values
        #hour = np.clip(hour, 0, 23)
        #minute = np.clip(minute, 0, 59)
        #minutes_of_day = hour * 60 + minute  # 0..1439
        #tid_data = (minutes_of_day // step_minutes).astype(int)  # 0..71 (20min)
        #tid_data = np.clip(tid_data, 0, slots_per_day - 1)

        # ===== 生成序列 =====
        X_train_list, y_train_list, Time_train_emb, y_train_elePrice = [], [], [], []
        X_test_list, y_test_list, Time_test_emb, Time_stamp_list, y_test_elePrice = [], [], [], [], []

        indices = self.data.index
        curr_X, curr_y, curr_T, curr_Price = [], [], [], []
        for i in range(0, len(indices) - seq_len - pred_len + 1):
            curr_X.append(X_data[indices[i:i + seq_len]])
            #curr_T.append(T_data[indices[i:i + seq_len]])
            y_slice = slice(i + seq_len, i + seq_len + pred_len)
            y_idx = indices[i + seq_len: i+seq_len+pred_len]  # 目标点在全局data中的索引
            curr_y.append(y_data[y_idx])
            curr_T.append(T_data[y_idx])
            curr_Price.append(X_data[y_idx])
            Time_stamp_list.append(time_series[y_slice])

        n_train = int(len(curr_X) * train_ratio)

        X_train_list.extend(curr_X[:n_train])
        y_train_list.extend(curr_y[:n_train])
        Time_train_emb.extend(curr_T[:n_train])
        y_train_elePrice.extend(curr_Price[:n_train])

        X_test_list.extend(curr_X[n_train:])
        y_test_list.extend(curr_y[n_train:])
        Time_test_emb.extend(curr_T[n_train:])
        y_test_elePrice.extend(curr_Price[:n_train])

        self.train_time = np.array(Time_stamp_list[:n_train])
        self.test_time = np.array(Time_stamp_list[n_train:])

        return (np.array(X_train_list), np.array(y_train_list), np.array(Time_train_emb), np.array(y_train_elePrice),
                np.array(X_test_list), np.array(y_test_list), np.array(Time_test_emb), np.array(y_test_elePrice))


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

    X_train, y_train, Time_train_emb, y_train_elePrice, X_test, y_test, Time_test_emb, y_test_elePrice = next_data.create_stratified_sequences()
    print(Time_train_emb)
