"""
Train and evaluate a PatchTST-based 1-hour module-demand forecaster.
训练并评估 1 小时模块需求预测模型。

This script is designed to run from the scripts/ folder.
本脚本设计为放在 scripts/ 文件夹下运行。

Running outputs are also stored under scripts/.
运行输出也统一保存在 scripts/ 文件夹下。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# =============================================================================
# 0. Path settings
# 0. 路径设置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Make both repository root and scripts folder importable.
# 同时加入项目根目录和 scripts 目录，方便导入自定义模块。
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =============================================================================
# 1. Custom project modules
# 1. 自定义项目模块
# =============================================================================

from Train_TSTRandomfor1Hours import Train_model
from Eval_TSTRandomfor1Hours import Pred_model
from scripts.Utils.MyGeneFeature_t_1h import mydefine
from scripts.Utils.utils_1h import build_loaders
from scripts.Utils.metric import NBLoss, TimeAwareAsymmetricLoss

try:
    from scripts.models.PatchTST_model_1h import PatchTST_WithSideInfo
except ModuleNotFoundError:
    from models.PatchTST_model_1h import PatchTST_WithSideInfo


# =============================================================================
# 2. Default configuration
# 2. 默认参数
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Input files / 输入文件
    # data_file is resolved relative to repository root first.
    # data_file 优先按项目根目录解析。
    "data_file": "data/sample/sample_station_6months.xlsx",

    # station_file is resolved relative to scripts/ first.
    # station_file 优先按 scripts/ 文件夹解析。
    "station_file": "station_list.xlsx",

    "station_code": "station_18",

    # Target variable used by the legacy pipeline.
    # 旧训练流程使用的目标变量名。
    "target_col_name": "num_modular",

    # Sequence settings / 序列设置
    "seq_length": 72,
    "predict_len": 3,
    "y_precise": [0],
    "limit" : 200,

    # Training settings / 训练设置
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 5e-4,

    # Model settings / 模型参数
    "side_indices": [4, 5, 11, 12, 13],
    "patch_len": 24,
    "stride": 12,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 3,
    "ff_dim": 256,
    "dropout": 0.3,

    # Output folders under scripts/
    # 输出文件夹统一放在 scripts/ 下。
    "result_dir": "PredictResult",
    "model_dir": "Model_State_dict",
    "prepared_data_dir": "outputs/_prepared_data",

    # Optional fallback if station_list.xlsx is incomplete.
    # 如果 station_list.xlsx 中没有模块数，可以在这里手动指定。
    "installed_modules": None,
}


# =============================================================================
# 3. Path utilities
# 3. 路径工具函数
# =============================================================================

def resolve_existing_path(
    path_value: Union[str, Path],
    base_dirs: List[Path],
) -> Path:
    """
    Resolve a file path by checking several base folders.
    通过多个候选基础路径解析文件路径。

    If the path is absolute, return it directly.
    如果是绝对路径，直接返回。
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate

    # If the file does not exist yet, return the first candidate.
    # 如果文件暂时不存在，则返回第一个候选路径，方便后续报错定位。
    return base_dirs[0] / path


def resolve_data_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve data path.
    解析数据路径。

    Priority:
    1. repository root
    2. scripts folder
    """
    return resolve_existing_path(path_value, [REPO_ROOT, SCRIPT_DIR])


def resolve_script_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve path under scripts/.
    解析 scripts/ 下的路径。

    This is used for station list and running outputs.
    用于 station_list 和运行输出。
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SCRIPT_DIR / path


def resolve_config_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve external config path.
    解析外部参数文件路径。

    Priority:
    1. current working directory
    2. scripts folder
    3. repository root
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return resolve_existing_path(path, [Path.cwd(), SCRIPT_DIR, REPO_ROOT])


# =============================================================================
# 4. Config parsing
# 4. 参数解析
# =============================================================================

def load_json_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load external JSON configuration.
    读取外部 JSON 参数文件。
    """
    if config_path is None:
        return {}

    path = resolve_config_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f">>> Loaded config file: {path}")
    return config


def parse_args() -> Dict[str, Any]:
    """
    Parse command-line arguments and merge with defaults/config file.
    解析命令行参数，并与默认参数/外部参数文件合并。
    """
    parser = argparse.ArgumentParser(
        description="Train PatchTST model for 1-hour station-level module-demand forecasting."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to external JSON configuration file. 外部 JSON 参数文件路径。",
    )

    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--station-file", type=str, default=None)
    parser.add_argument("--station-code", type=str, default=None)
    parser.add_argument("--target-col-name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--predict-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    parser.add_argument(
        "--y-precise",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizon index list. Example: --y-precise 0",
    )

    parser.add_argument(
        "--installed-modules",
        type=float,
        default=None,
        help="Optional installed module number. 可选：站点模块数。",
    )

    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--prepared-data-dir", type=str, default=None)

    args = parser.parse_args()

    # Default -> external config -> command-line override.
    # 默认参数 -> 外部参数文件 -> 命令行覆盖。
    final_config = DEFAULT_CONFIG.copy()

    file_config = load_json_config(args.config)
    final_config.update(file_config)

    cli_args = vars(args)
    for key, value in cli_args.items():
        if key == "config":
            continue
        if value is not None:
            final_config[key] = value

    return final_config


# =============================================================================
# 5. Data preparation
# 5. 数据准备
# =============================================================================

def standardize_sample_for_legacy_pipeline(
    df: pd.DataFrame,
    station_code: str = "station_18",
) -> pd.DataFrame:
    """
    Convert public sample schema to the legacy training schema.

    将公开样例数据的英文字段转换为旧训练流程需要的字段。

    This does not change the public sample file.
    It only prepares a temporary parquet file for the existing pipeline.
    不修改公开样例文件，只生成临时 parquet 给旧流程使用。
    """
    df = df.copy()

    column_map = {
        "module_demand": "num_modular",
        "active_vehicle_count": "num_charging",
        "arriving_vehicle_count": "num_coming",
        "aggregate_power_kw": "Power_kW",
        "electricity_price": "electric_price",
        "raw_hour": "hour",
        "raw_minute": "minute",
        "day_type": "day_type_label",
    }

    for public_col, legacy_col in column_map.items():
        if public_col in df.columns and legacy_col not in df.columns:
            df[legacy_col] = df[public_col]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        if "hour" not in df.columns:
            df["hour"] = df["time"].dt.hour

        if "minute" not in df.columns:
            df["minute"] = df["time"].dt.minute

    if "FID" not in df.columns:
        df["FID"] = station_code

    if "FID_name" not in df.columns:
        df["FID_name"] = "sample_station"

    required_cols = [
        "time",
        "num_modular",
        "num_charging",
        "num_coming",
        "Power_kW",
        "electric_price",
        "hour",
        "minute",
        "day_type_label",
        "FID",
        "FID_name",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise KeyError(
            "The following legacy columns are still missing after conversion: "
            f"{missing_cols}"
        )

    return df


def read_input_file(data_file: Path) -> pd.DataFrame:
    """
    Read CSV, Excel or parquet file.
    读取 CSV、Excel 或 parquet 文件。
    """
    suffix = data_file.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(data_file)

    if suffix == ".csv":
        return pd.read_csv(data_file)

    if suffix == ".xlsx":
        return pd.read_excel(data_file, engine="openpyxl")

    if suffix == ".xls":
        return pd.read_excel(data_file)

    raise ValueError(f"Unsupported data file type: {suffix}")


def prepare_input_file(
    data_file: Path,
    station_code: str,
    prepared_data_dir: Path,
) -> Path:
    """
    Prepare input file for the legacy sequence builder.

    读取公开样例数据，转换为旧字段结构，并保存到 scripts/outputs/_prepared_data/。
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print("=" * 80)
    print(f">>> Reading input file: {data_file}")
    print(f">>> File suffix: {data_file.suffix.lower()}")

    df = read_input_file(data_file)

    print(f">>> Loaded shape: {df.shape}")
    print(">>> Columns before legacy conversion:")
    print(df.columns.tolist())

    df = standardize_sample_for_legacy_pipeline(
        df=df,
        station_code=station_code,
    )

    print(">>> Columns after legacy conversion:")
    print(df.columns.tolist())

    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    temp_parquet = prepared_data_dir / f"{data_file.stem}_legacy.parquet"
    df.to_parquet(temp_parquet, index=False)

    print(f">>> Converted input file to temporary parquet: {temp_parquet}")

    return temp_parquet


# =============================================================================
# 6. Station metadata
# 6. 站点信息
# =============================================================================

def find_first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Find the first existing column from candidate names.
    从候选列名中找到第一个存在的列。
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_modules_from_sample(
    prepared_data_file: Path,
    target_col_name: str,
) -> np.ndarray:
    """
    Infer installed modules from sample data if station list cannot be matched.

    如果 station_list 无法匹配，则从样例数据中推断模块数。
    This is only a fallback for sample workflow demonstration.
    该逻辑仅用于样例流程演示。
    """
    df = pd.read_parquet(prepared_data_file)

    if "installed_modules" in df.columns:
        modules = int(df["installed_modules"].dropna().max())
        print(f">>> Installed modules inferred from installed_modules: {modules}")
        return np.array([modules])

    if target_col_name in df.columns:
        modules = int(df[target_col_name].dropna().max())
        print(f">>> Installed modules inferred from max({target_col_name}): {modules}")
        print(
            ">>> Note: this fallback is for sample demonstration only. "
            "For formal analysis, use the actual installed module number."
        )
        return np.array([modules])

    raise ValueError(
        "Cannot infer installed modules from sample data. "
        f"Target column not found: {target_col_name}"
    )


def load_station_modules(
    station_file: Path,
    station_code: str,
    installed_modules_fallback: Optional[float],
    prepared_data_file: Path,
    target_col_name: str,
) -> np.ndarray:
    """
    Load installed module number from station_list.xlsx.

    从 station_list.xlsx 读取站点模块数。
    If not available, use config fallback or infer from sample data.
    如果不可用，则使用配置兜底或从样例数据推断。
    """
    station_code_candidates = [
        "station_code",
        "station_id",
        "station_id_anonymized",
        "电站编号",
        "站点编号",
    ]

    module_col_candidates = [
        "installed_modules",
        "max_modules",
        "module_count",
        "站点模块数",
        "模块数",
    ]

    if station_file.exists():
        station_df = pd.read_excel(station_file)

        print("=" * 80)
        print(f">>> Reading station file: {station_file}")
        print(">>> Station-list columns:")
        print(station_df.columns.tolist())

        station_code_col = find_first_existing_column(
            station_df,
            station_code_candidates,
        )

        module_col = find_first_existing_column(
            station_df,
            module_col_candidates,
        )

        print(f">>> Matched station-code column: {station_code_col}")
        print(f">>> Matched module-count column: {module_col}")

        if station_code_col is not None and module_col is not None:
            station_df[station_code_col] = station_df[station_code_col].astype(str)

            station_row = station_df[
                station_df[station_code_col] == str(station_code)
            ]

            if not station_row.empty:
                modules = station_row[module_col].values
                print(f">>> Installed modules loaded from station file: {modules}")
                return modules

            print(f">>> Station code not found in station file: {station_code}")

        else:
            print(">>> Station file found, but required columns were not matched.")

    else:
        print(f">>> Station file not found: {station_file}")

    if installed_modules_fallback is not None:
        modules = np.array([installed_modules_fallback])
        print(f">>> Installed modules loaded from config: {modules}")
        return modules

    return infer_modules_from_sample(
        prepared_data_file=prepared_data_file,
        target_col_name=target_col_name,
    )


# =============================================================================
# 7. Training and prediction
# 7. 训练与预测
# =============================================================================

def train_and_predict_station(config: Dict[str, Any]) -> None:
    """
    Train the model and export prediction results for one station.
    对单个站点训练模型并导出预测结果。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    station_code = str(config["station_code"])

    # Input data: repository root first.
    # 输入数据：优先从项目根目录解析。
    data_file = resolve_data_path(config["data_file"])

    # Station metadata and outputs: scripts/ folder.
    # 站点信息和运行输出：统一放在 scripts/ 下。
    station_file = resolve_script_path(config["station_file"])
    result_dir = resolve_script_path(config["result_dir"])
    model_dir = resolve_script_path(config["model_dir"])
    prepared_data_dir = resolve_script_path(config["prepared_data_dir"])

    # Create all output directories.
    # 创建所有输出目录。
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    prepared_data_file = prepare_input_file(
        data_file=data_file,
        station_code=station_code,
        prepared_data_dir=prepared_data_dir,
    )

    max_modules = load_station_modules(
        station_file=station_file,
        station_code=station_code,
        installed_modules_fallback=config.get("installed_modules"),
        prepared_data_file=prepared_data_file,
        target_col_name=config["target_col_name"],
    )

    csv_save_path = result_dir / f"reg_patchTST_1h_{station_code}.csv"
    model_save_path = (
        model_dir / f"best_model_1h_{station_code}_t{config['y_precise'][0]}.pth"
    )

    # Create parent folders again before external save calls.
    # 在外部训练/预测函数保存前再次确认目录存在。
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f">>> REPO_ROOT: {REPO_ROOT}")
    print(f">>> SCRIPT_DIR: {SCRIPT_DIR}")
    print(f">>> Device: {device}")
    print(f">>> Station code: {station_code}")
    print(f">>> Data file: {data_file}")
    print(f">>> Prepared data file: {prepared_data_file}")
    print(f">>> Station file: {station_file}")
    print(f">>> Prediction output: {csv_save_path}")
    print(f">>> Model checkpoint: {model_save_path}")
    print(f">>> Result dir exists: {result_dir.exists()} -> {result_dir}")
    print(f">>> Model dir exists: {model_dir.exists()} -> {model_dir}")
    print(f">>> Prepared data dir exists: {prepared_data_dir.exists()} -> {prepared_data_dir}")

    # -------------------------------------------------------------------------
    # 1. Build sequences
    # 1. 构建训练与测试序列
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(">>> Loading and preparing sequences...")

    dataset_builder = mydefine(
        str(prepared_data_file),
        config["target_col_name"],
        config["seq_length"],
        config["predict_len"],
        norm=False,
    )

    (
        X_train,
        y_train,
        time_train_emb,
        y_train_ele_price,
        X_test,
        y_test,
        time_test_emb,
        y_test_ele_price,
    ) = dataset_builder.create_stratified_sequences()

    print(
        "Train shape: "
        f"X={X_train.shape}, y={y_train.shape}, time={time_train_emb.shape}"
    )

    train_loader, test_loader = build_loaders(
        X_train,
        y_train,
        time_train_emb,
        y_train_ele_price,
        X_test,
        y_test,
        time_test_emb,
        y_test_ele_price,
        config["batch_size"],
    )

    n_vars = X_train.shape[2]

    print(f">>> Number of variables: {n_vars}")
    print(f">>> Side-variable indices: {config['side_indices']}")

    # -------------------------------------------------------------------------
    # 2. Build model
    # 2. 构建 PatchTST 模型
    # -------------------------------------------------------------------------
    model = PatchTST_WithSideInfo(
        seq_len=config["seq_length"],
        pred_len=len(config["y_precise"]),
        n_vars=n_vars,
        side_vars_indices=config["side_indices"],
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-3,
    )

    # Negative-binomial loss for discrete module demand.
    # 负二项损失用于刻画离散模块需求。
    criterion_nll = NBLoss()

    # Time-aware asymmetric loss for operational preferences.
    # 时间感知非对称损失用于体现运行偏好。
    criterion_asym_mae = TimeAwareAsymmetricLoss(
        night_penalty=1,
        morning_penalty=1,
    ).to(device)

    # -------------------------------------------------------------------------
    # 3. Train model
    # 3. 训练模型
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(">>> Training model...")

    Train_model(
        model,
        train_loader,
        optimizer,
        criterion_asym_mae,
        criterion_nll,
        config["epochs"],
        device,
        test_loader,
        max_modules,
        config["y_precise"],
        str(model_save_path),
    )

    # -------------------------------------------------------------------------
    # 4. Predict and save results
    # 4. 预测并保存结果
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(">>> Generating predictions...")

    # Ensure folders still exist before prediction output.
    # 预测输出前再次确认目录存在。
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    Pred_model(
        dataset_builder,
        model,
        train_loader,
        test_loader,
        device,
        str(csv_save_path),
        config["limit"],
        max_modules,
        config["y_precise"],
        str(model_save_path),
        station_code,
    )

    print("=" * 80)
    print(f">>> Prediction saved to: {csv_save_path}")
    print(f">>> Model saved to: {model_save_path}")
    print("Done.")


# =============================================================================
# 8. Main entrance
# 8. 主程序入口
# =============================================================================

def main() -> None:
    """
    Main entrance.
    主程序入口。
    """
    config = parse_args()
    train_and_predict_station(config)


if __name__ == "__main__":
    main()