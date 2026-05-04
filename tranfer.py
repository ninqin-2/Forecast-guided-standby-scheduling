import pandas as pd
from pathlib import Path

# ============================================================
# 0. 参数设置
# ============================================================

SOURCE_STATION_CODE = "3101120225"

input_dir = Path(r"F:\代码库\TLD_modular_predict\resampled_20min")

matched_files = sorted(input_dir.glob(f"{SOURCE_STATION_CODE}_*.parquet"))

if not matched_files:
    raise FileNotFoundError(
        f"未找到站点 {SOURCE_STATION_CODE} 对应的 parquet 文件："
        f"{input_dir / (SOURCE_STATION_CODE + '_*.parquet')}"
    )

if len(matched_files) > 1:
    print("=" * 80)
    print("Warning: 找到多个匹配文件，将使用第一个：")
    for f in matched_files:
        print(f" - {f}")

input_path = matched_files[0]

print("=" * 80)
print(f"Using input file: {input_path}")

output_dir = Path(r"F:\代码库\forecast-guided-standby-scheduling\data\sample")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "sample_station_6months.csv"
xlsx_path = output_dir / "sample_station_6months.xlsx"

ANON_STATION_ID = "S001"
ANON_STATION_NAME = "sample_station"

EXPORT_CSV = True
EXPORT_XLSX = True

# ============================================================
# 1. 读取 parquet
# ============================================================

df = pd.read_parquet(input_path)

print("=" * 80)
print("Original shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

original_columns = df.columns.tolist()

# ============================================================
# 2. 时间列处理
# ============================================================

time_col = "time"

if time_col not in df.columns:
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    if len(datetime_cols) > 0:
        time_col = datetime_cols[0]
    else:
        raise ValueError("未找到时间列。请确认数据中存在 `time` 列。")

print("=" * 80)
print(f"Detected time column: {time_col}")

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).copy()
df = df.sort_values(time_col).reset_index(drop=True)

print("Full time range:", df[time_col].min(), "to", df[time_col].max())

# ============================================================
# 3. 保留最新 6 个月
# ============================================================

end_time = df[time_col].max()
start_time = end_time - pd.DateOffset(months=6)

sample_df = df[
    (df[time_col] >= start_time) &
    (df[time_col] <= end_time)
].copy()

sample_df = sample_df.sort_values(time_col).reset_index(drop=True)

print("=" * 80)
print("Sample time range:", sample_df[time_col].min(), "to", sample_df[time_col].max())
print("Sample shape:", sample_df.shape)

# ============================================================
# 4. 原位匿名化敏感站点字段
#    不删除、不新增、不移动列
# ============================================================

if "FID" in sample_df.columns:
    sample_df["FID"] = ANON_STATION_ID

if "FID_name" in sample_df.columns:
    sample_df["FID_name"] = ANON_STATION_NAME

# ============================================================
# 5. day_type_label 内容英文化
#    只替换内容，不改变列位置
# ============================================================

day_type_map = {
    "工作日": "Weekday",
    "周末": "Weekend",
    "双休日": "Weekend",
    "节假日": "Holiday",
    "法定节假日": "Statutory Holiday",
    "假日": "Holiday",
    "调休日": "Adjusted Workday",
    "补班": "Adjusted Workday",
    "补工作日": "Adjusted Workday",
    "调休补班": "Adjusted Workday",
    "休息日": "Rest Day",
    "普通日": "Regular Day",

    "weekday": "Weekday",
    "weekend": "Weekend",
    "holiday": "Holiday",
    "statutory holiday": "Statutory Holiday",
    "adjusted workday": "Adjusted Workday",
    "rest day": "Rest Day",
    "regular day": "Regular Day",
}

def clean_day_type(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .replace(day_type_map)
    )

if "day_type_label" in sample_df.columns:
    print("=" * 80)
    print("day_type_label before mapping:")
    print(sample_df["day_type_label"].dropna().unique())

    sample_df["day_type_label"] = clean_day_type(sample_df["day_type_label"])

    print("day_type_label after mapping:")
    print(sample_df["day_type_label"].dropna().unique())

elif "day_type" in sample_df.columns:
    print("=" * 80)
    print("day_type before mapping:")
    print(sample_df["day_type"].dropna().unique())

    sample_df["day_type"] = clean_day_type(sample_df["day_type"])

    print("day_type after mapping:")
    print(sample_df["day_type"].dropna().unique())

# ============================================================
# 6. 列名英文化
#    只 rename，不 reorder
# ============================================================

rename_map = {
    "num_modular": "module_demand",
    "num_charging": "active_vehicle_count",
    "num_coming": "arriving_vehicle_count",
    "Power_kW": "aggregate_power_kw",
    "electric_price": "electricity_price",
    "hour": "raw_hour",
    "minute": "raw_minute",
    "FID": "station_id_anonymized",
    "FID_name": "station_name_anonymized",
}

# 构造“原始列顺序对应的英文列顺序”
expected_english_columns = [
    rename_map.get(col, col)
    for col in original_columns
    if col in sample_df.columns
]

sample_df = sample_df.rename(columns=rename_map)

# ============================================================
# 7. 强制恢复原始列顺序
# ============================================================

missing_after_rename = [
    col for col in expected_english_columns
    if col not in sample_df.columns
]

extra_after_rename = [
    col for col in sample_df.columns
    if col not in expected_english_columns
]

if missing_after_rename:
    raise ValueError(f"Missing columns after renaming: {missing_after_rename}")

if extra_after_rename:
    print("=" * 80)
    print("Warning: extra columns will be kept at the end:")
    print(extra_after_rename)

sample_df = sample_df[
    expected_english_columns + extra_after_rename
].copy()

# ============================================================
# 8. 列顺序检查
# ============================================================

print("=" * 80)
print("Column order after English conversion:")
for i, col in enumerate(sample_df.columns, start=1):
    print(f"{i:02d}. {col}")

print("=" * 80)
print("Original column count:", len(original_columns))
print("Output column count:", len(sample_df.columns))

if len(sample_df.columns) != len(original_columns):
    print("Warning: column count changed.")
else:
    print("Column count unchanged.")

# ============================================================
# 9. 导出
# ============================================================

if EXPORT_CSV:
    sample_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("=" * 80)
    print("Saved CSV:")
    print(csv_path)

if EXPORT_XLSX:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="sample_6months")

    print("=" * 80)
    print("Saved XLSX:")
    print(xlsx_path)

# ============================================================
# 10. 最终检查
# ============================================================

print("=" * 80)
print("Final sample shape:", sample_df.shape)
print("Preview:")
print(sample_df.head())