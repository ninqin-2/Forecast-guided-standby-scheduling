"""
Run multi-step module-demand prediction and uncertainty adjustment.

运行三阶段模块需求预测，并进行高波动时段修正。

This script is prepared for the public GitHub repository.
It uses anonymized sample data instead of raw, non-anonymized station data.

本脚本用于公开仓库版本。
输入为匿名 sample 数据，不再调用未匿名的原始站点数据。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 0. Path settings
# 0. 路径设置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

for p in [REPO_ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# =============================================================================
# 1. Custom project modules
# 1. 自定义项目模块
# =============================================================================

from Train_TSTRandomfor1Hours import Train_model
from Eval_TSTRandomSimple import Pred_model
from scripts.Utils.MyGeneFeature_t_1h import mydefine
from scripts.Utils.utils_1h import *
from scripts.Utils.metric import NBLoss, TimeAwareAsymmetricLoss
from scripts.Utils.MultiStepDeal import *
from ErrorDist import MyErrorDistDeal

try:
    from scripts.models.PatchTST_model_1h import PatchTST_WithSideInfo
except ModuleNotFoundError:
    from models.PatchTST_model_1h import PatchTST_WithSideInfo


# =============================================================================
# 2. Default configuration
# 2. 默认参数
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Input data / 输入数据
    "data_file": "data/sample/sample_station_6months.xlsx",
    "station_file": "station_list.xlsx",
    "station_code": "station_18",

    # Target variable used by legacy internal pipeline
    # 旧训练流程内部使用的目标字段
    "target_col_name": "num_modular",

    # Sequence settings / 序列设置
    "seq_length": 72,
    "predict_len": 30,
    "batch_size": 64,

    # Training settings / 训练设置
    "epochs": 10,
    "learning_rate": 5e-4,
    "is_training": True,

    # Uncertainty adjustment / 不确定性修正
    "eps": 0.05,

    # Model settings / 模型参数
    "side_indices": [4, 5, 11, 12, 13],
    "patch_len": 24,
    "stride": 12,
    "d_model_default": 128,
    "d_model_by_horizon": {
        "0": 128,
        "1": 128,
        "2": 256
    },
    "n_heads": 8,
    "n_layers": 3,
    "ff_dim": 256,
    "dropout": 0.3,

    # Forecast horizons / 三个预测阶段
    "horizons": [0, 1, 2],

    # Output folders under scripts/
    # 输出统一放在 scripts/ 文件夹下
    "result_dir": "PredictResult",
    "model_dir": "Model_State_dict",
    "figure_dir": "Figures",

    # Plot setting / 绘图设置
    "xlim_set": [200, 600],

    # Optional fallback when station_list.xlsx is unavailable
    # station_list.xlsx 缺失时的兜底模块数
    "installed_modules": None
}


# =============================================================================
# 3. Path helpers
# 3. 路径工具
# =============================================================================

def resolve_existing_path(
    path_value: Union[str, Path],
    base_dirs: List[Path],
) -> Path:
    """
    Resolve a path by checking several base folders.
    按多个基础路径解析文件。
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate

    return base_dirs[0] / path


def resolve_data_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve data file path.
    解析数据文件路径。
    """
    return resolve_existing_path(path_value, [REPO_ROOT, SCRIPT_DIR])


def resolve_script_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve path under scripts/.
    解析 scripts/ 下的路径。
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SCRIPT_DIR / path


def resolve_config_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve external config path.
    解析外部配置文件路径。
    """
    return resolve_existing_path(path_value, [Path.cwd(), SCRIPT_DIR, REPO_ROOT])


# =============================================================================
# 4. Config parsing
# 4. 参数解析
# =============================================================================

def load_json_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load JSON config.
    读取 JSON 配置文件。
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
    Parse command-line arguments.
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Run multi-step module-demand prediction on anonymized sample data."
    )

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--station-file", type=str, default=None)
    parser.add_argument("--station-code", type=str, default=None)
    parser.add_argument("--target-col-name", type=str, default=None)

    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--predict-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--is-training",
        type=int,
        choices=[0, 1],
        default=None,
        help="1 = train models before prediction; 0 = load existing checkpoints."
    )

    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--installed-modules", type=float, default=None)

    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update(load_json_config(args.config))

    cli_args = vars(args)
    for key, value in cli_args.items():
        if key == "config":
            continue
        if value is not None:
            if key == "is_training":
                config[key] = bool(value)
            else:
                config[key] = value

    return config


# =============================================================================
# 5. Data and station metadata
# 5. 数据与站点信息
# =============================================================================

def read_input_table(path: Path) -> pd.DataFrame:
    """
    Read parquet, csv or Excel file.
    读取 parquet、csv 或 Excel。
    """
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported data file type: {suffix}")


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Find first matched column name.
    查找第一个匹配列名。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_modules_from_data(
    data_file: Path,
    target_col_name: str,
) -> np.ndarray:
    """
    Infer installed modules from sample data as fallback.
    从 sample 数据推断模块数，仅作为兜底。
    """
    df = read_input_table(data_file)

    if "installed_modules" in df.columns:
        modules = int(df["installed_modules"].dropna().max())
        print(f">>> Installed modules inferred from installed_modules: {modules}")
        return np.array([modules])

    candidate_cols = [
        target_col_name,
        "num_modular",
        "module_demand"
    ]

    for c in candidate_cols:
        if c in df.columns:
            modules = int(pd.to_numeric(df[c], errors="coerce").dropna().max())
            print(f">>> Installed modules inferred from max({c}): {modules}")
            print(">>> Note: this fallback is only for sample workflow demonstration.")
            return np.array([modules])

    raise ValueError(
        "Installed module number was not found. "
        "Please provide station_list.xlsx or installed_modules."
    )


def load_station_modules(
    station_file: Path,
    station_code: str,
    data_file: Path,
    target_col_name: str,
    installed_modules_fallback: Optional[float] = None,
) -> np.ndarray:
    """
    Load installed module number from station_list.xlsx.
    从 station_list.xlsx 读取站点模块数。
    """
    if station_file.exists():
        station_df = pd.read_excel(station_file)

        print("=" * 80)
        print(f">>> Reading station file: {station_file}")
        print(">>> Station-list columns:")
        print(station_df.columns.tolist())

        station_code_col = find_first_existing_column(
            station_df,
            [
                "station_code",
                "station_id",
                "station_id_anonymized",
                "电站编号",
                "站点编号"
            ],
        )

        module_col = find_first_existing_column(
            station_df,
            [
                "installed_modules",
                "max_modules",
                "module_count",
                "站点模块数",
                "模块数"
            ],
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
        print(f">>> Station file not found: {station_file}")

    if installed_modules_fallback is not None:
        modules = np.array([installed_modules_fallback])
        print(f">>> Installed modules loaded from config: {modules}")
        return modules

    return infer_modules_from_data(data_file, target_col_name)


# =============================================================================
# 6. Training and prediction helpers
# 6. 训练与预测工具函数
# =============================================================================

def build_model(
    horizon: int,
    n_vars: int,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build PatchTST model for a given horizon.
    构建指定预测步长的 PatchTST 模型。
    """
    d_model_map = config.get("d_model_by_horizon", {})
    d_model = d_model_map.get(str(horizon), config["d_model_default"])

    model = PatchTST_WithSideInfo(
        seq_len=config["seq_length"],
        pred_len=1,
        n_vars=n_vars,
        side_vars_indices=config["side_indices"],
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=d_model,
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
    ).to(device)

    return model


def train_one_horizon(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    horizon: int,
    max_modules: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    model_save_path: Path,
) -> None:
    """
    Train model for one prediction horizon.
    训练单个预测步长模型。
    """
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-3,
    )

    criterion_nll = NBLoss()
    criterion_asym_mae = TimeAwareAsymmetricLoss(
        night_penalty=1,
        morning_penalty=1,
    ).to(device)

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
        [horizon],
        str(model_save_path),
    )


def predict_one_horizon(
    next_data,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    horizon: int,
    max_modules: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    csv_save_path: Path,
    model_save_path: Path,
    station_code: str,
    test_save: bool = True,
):
    """
    Run prediction for one horizon.
    对单个预测步长进行预测。
    """
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    return Pred_model(
        next_data,
        model,
        train_loader,
        test_loader,
        device,
        str(csv_save_path),
        max_modules,
        [horizon],
        str(model_save_path),
        station_code,
        test_save=test_save,
    )


# =============================================================================
# 7. Main workflow
# 7. 主流程
# =============================================================================

def run_multi_step_plan(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run three-stage prediction and uncertainty adjustment.
    运行三阶段预测与不确定性修正。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    station_code = str(config["station_code"])
    data_file = resolve_data_path(config["data_file"])
    station_file = resolve_script_path(config["station_file"])

    result_dir = resolve_script_path(config["result_dir"])
    model_dir = resolve_script_path(config["model_dir"])
    figure_dir = resolve_script_path(config["figure_dir"])

    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print("=" * 80)
    print(f">>> REPO_ROOT: {REPO_ROOT}")
    print(f">>> SCRIPT_DIR: {SCRIPT_DIR}")
    print(f">>> Device: {device}")
    print(f">>> Station code: {station_code}")
    print(f">>> Data file: {data_file}")
    print(f">>> Station file: {station_file}")
    print(f">>> Result dir: {result_dir}")
    print(f">>> Model dir: {model_dir}")
    print(f">>> Figure dir: {figure_dir}")

    max_modules = load_station_modules(
        station_file=station_file,
        station_code=station_code,
        data_file=data_file,
        target_col_name=config["target_col_name"],
        installed_modules_fallback=config.get("installed_modules"),
    )

    # -------------------------------------------------------------------------
    # 1. Data preparation
    # 1. 数据准备
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(">>> Loading anonymized sample data...")

    next_data = mydefine(
        str(data_file),
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
    ) = next_data.create_stratified_sequences()

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
    # 2. Multi-horizon prediction
    # 2. 多步预测
    # -------------------------------------------------------------------------
    train_dist_df = None

    for horizon in config["horizons"]:
        print("=" * 80)
        print(f">>> Horizon t+{horizon}")

        csv_save_path = result_dir / f"reg_patchTST_1h_{station_code}_t{horizon}.csv"
        model_save_path = model_dir / f"best_model_1h_{station_code}_t{horizon}.pth"

        model = build_model(
            horizon=horizon,
            n_vars=n_vars,
            config=config,
            device=device,
        )

        if config["is_training"]:
            train_one_horizon(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                horizon=horizon,
                max_modules=max_modules,
                config=config,
                device=device,
                model_save_path=model_save_path,
            )

        predict_one_horizon(
            next_data=next_data,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            horizon=horizon,
            max_modules=max_modules,
            config=config,
            device=device,
            csv_save_path=csv_save_path,
            model_save_path=model_save_path,
            station_code=station_code,
            test_save=True,
        )

        # Use training distribution from horizon 0 for residual adjustment.
        # 使用 t+0 的训练分布做误差修正。
        if horizon == 0:
            train_dist_df = predict_one_horizon(
                next_data=next_data,
                model=model,
                train_loader=train_loader,
                test_loader=train_loader,
                horizon=horizon,
                max_modules=max_modules,
                config=config,
                device=device,
                csv_save_path=csv_save_path,
                model_save_path=model_save_path,
                station_code=station_code,
                test_save=False,
            )

    if train_dist_df is None:
        raise RuntimeError("Training distribution dataframe was not generated.")

    # -------------------------------------------------------------------------
    # 3. High-volatility adjustment
    # 3. 高波动时段修正
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(">>> Combining horizon predictions...")

    collect = []

    for horizon in config["horizons"]:
        csv_path = result_dir / f"reg_patchTST_1h_{station_code}_t{horizon}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {csv_path}")

        df_pred = pd.read_csv(csv_path)
        df_pred["Time"] = pd.to_datetime(df_pred["Time"])

        if horizon == 0:
            temp = df_pred.copy()
            temp = temp.rename(
                columns={
                    "True": f"true_{horizon}",
                    "Pred_Mu": f"t_{horizon}",
                }
            )
        else:
            temp = df_pred[["Time", "True", "Pred_Mu"]].copy()
            temp = temp.rename(
                columns={
                    "True": f"true_{horizon}",
                    "Pred_Mu": f"t_{horizon}",
                }
            )

        collect.append(temp)

    out_df = MyErrorDistDeal(
        collect,
        train_dist_df,
        config["eps"],
        state=True,
    )

    combined_path = result_dir / f"reg_patchTST_1h_{station_code}.xlsx"
    out_df.to_excel(combined_path, index=False)

    print(f">>> Combined prediction saved to: {combined_path}")

    # -------------------------------------------------------------------------
    # 4. Evaluation and plotting
    # 4. 效果评估与绘图
    # -------------------------------------------------------------------------
    true_values = out_df["true_0"].values
    pred_values = out_df["t_0"].values
    optimized_pred_values = out_df["m_final"].values

    error_statistics = compute_error_statistics(
        true_values,
        pred_values,
        optimized_pred_values,
    )

    print("=" * 80)
    print("Error statistics before and after adjustment:")
    for key, value in error_statistics.items():
        print(f"{key}: {value}")

    figure_path = figure_dir / f"module_plan_{station_code}.png"

    PlotModulesOpenPlan(
        config["eps"],
        max_modules,
        config["xlim_set"],
        out_df,
        savepath=str(figure_path),
    )

    print(f">>> Figure saved to: {figure_path}")

    return error_statistics


# =============================================================================
# 8. Main entrance
# 8. 主程序入口
# =============================================================================

def main() -> None:
    config = parse_args()
    run_multi_step_plan(config)
    print("Done.")


if __name__ == "__main__":
    main()