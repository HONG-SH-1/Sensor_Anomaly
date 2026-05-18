"""Build notebooks/clean/visualization_only.ipynb from all_clean viz cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_ALL = json.loads((ROOT / "notebooks/clean/all_clean.ipynb").read_text(encoding="utf-8"))
VIZ_IDX = [8, 10, 12, 14, 16, 19, 22, 28, 32, 35]
VIZ_NAMES = [
    "01_missing_labels",
    "02_all_sensors",
    "03_ks_distribution",
    "04_adf",
    "05_correlation",
    "06_vif",
    "07_window_dist",
    "08_model_comparison",
    "09_roc",
    "10_optuna",
    "11_final_result",
]

SETUP = r'''# Visualization only — checkpoints + sensor.csv (no training / Optuna / Gradio)
import os
import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import platform
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

_cwd = Path.cwd().resolve()
REPO_ROOT = _cwd if (_cwd / 'data' / 'raw' / 'sensor.csv').is_file() else _cwd.parents[1]
os.chdir(REPO_ROOT)

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    fm._load_fontmanager(try_read_cache=False)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = str(REPO_ROOT / 'data' / 'raw' / 'sensor.csv')
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f'sensor.csv 없음: {DATA_PATH}')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_SIZE = 50
STEP_SIZE = 10
THRESHOLD_PCT = 95
MISSING_THRESH = 0.50
DEFAULT_HIDDEN = 64
DEFAULT_LATENT = 32
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.1
plot_colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

print(f'REPO_ROOT: {REPO_ROOT}')
print(f'DATA_PATH: {DATA_PATH}')
print(f'Device: {device}')
'''


def cell_md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": lines}


def patch_optuna_cell(src: str) -> str:
    src = src.replace(
        "trial_vals = [t.value for t in study.trials if t.value is not None]",
        "# trial_vals: prepare_variant()가 체크포인트 .log에서 파싱",
    )
    src = src.replace(
        "axes[0].axhline(study.best_value, color='#EF4444', linestyle='--',\n"
        "                label=f'최적 F2={study.best_value:.4f}')",
        "best_f2 = study_best_value if study_best_value is not None else (max(trial_vals) if trial_vals else 0)\n"
        "axes[0].axhline(best_f2, color='#EF4444', linestyle='--',\n"
        "                label=f'최적 F2={best_f2:.4f}')",
    )
    src = src.replace(
        "    imp = optuna.importance.get_param_importances(study)",
        "    imp = {}",
    )
    return src


def patch_adf_cell(src: str) -> str:
    old = """adf_df = pd.DataFrame(adf_res).T
n_s = adf_df['stationary'].sum()

fig, ax = plt.subplots(figsize=(12, 4))"""
    new = """if not adf_res:
    print('ADF: 검정 가능한 센서가 없습니다 (04_adf.png 생략).')
else:
    adf_df = pd.DataFrame(adf_res).T
    if 'stationary' not in adf_df.columns:
        adf_df['stationary'] = False
    n_s = int(adf_df['stationary'].sum())

    fig, ax = plt.subplots(figsize=(12, 4))"""
    if old not in src:
        return src
    src = src.replace(old, new)
    src = src.replace(
        "non_stationary = adf_df[adf_df['stationary'] == False].index.tolist()",
        "    non_stationary = adf_df[~adf_df['stationary'].astype(bool)].index.tolist()",
    )
    # indent plot block inside else
    lines = src.splitlines(keepends=True)
    out, in_else = [], False
    for line in lines:
        if line.startswith("else:"):
            in_else = True
            out.append(line)
            continue
        if in_else and line.startswith("non_stationary"):
            in_else = False
        if in_else and line.strip() and not line.startswith("    "):
            if not line.startswith("if not adf_res"):
                line = "    " + line
        out.append(line)
    return "".join(out)


def patch_window_cell(src: str) -> str:
    marker = "train_loader_default = DataLoader"
    if marker in src:
        src = src[: src.index(marker)] + (
            "# (visualization_only) 학습 로더 생략\n"
            "print('Window tensors ready for checkpoint eval.')\n"
        )
    return src


def main():
    models_utils = (ROOT / "scripts/_viz_nb_utils.py").read_text(encoding="utf-8")
    load_data = """df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = df_raw.columns.str.strip()
ts_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
if ts_col:
    df_raw.rename(columns={ts_col: 'timestamp'}, inplace=True)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.set_index('timestamp').sort_index()
sensor_cols = [c for c in df_raw.columns if c.startswith('sensor')]
print(f'Shape: {df_raw.shape} | sensors: {len(sensor_cols)}')
"""

    cells = [
        cell_md(
            "# Visualization Only\n\n"
            "`all_clean` / `vif_clean`의 **plt.savefig 시각화 11종×2**만 재생성.\n\n"
            "- 데이터: `data/raw/sensor.csv`\n"
            "- 체크포인트: `models/checkpoints_all/`, `models/checkpoints_vif/`\n"
            "- 출력: `outputs/figures/all/`, `outputs/figures/vif/`\n"
            "- 학습·Optuna·Gradio 없음"
        ),
        cell_code(SETUP),
        cell_code(models_utils),
        cell_code(load_data),
    ]

    for variant, label in [("all", "ALL (51 sensors)"), ("vif", "VIF (~21 sensors)")]:
        cells.append(cell_md(f"---\n## {label}\n"))
        cells.append(cell_code(f"prepare_variant('{variant}')"))
        for idx, name in zip(VIZ_IDX, VIZ_NAMES):
            src = "".join(NB_ALL["cells"][idx]["source"])
            if idx == 14:
                src = patch_adf_cell(src)
            if idx == 22:
                src = patch_window_cell(src)
            if idx == 32:
                src = patch_optuna_cell(src)
            cells.append(cell_md(f"### {label} — {name}"))
            cells.append(cell_code(src))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    out = ROOT / "notebooks/clean/visualization_only.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
