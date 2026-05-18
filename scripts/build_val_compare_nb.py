"""Emit notebooks/val_compare/val_compare.ipynb — run from repo root: python scripts/build_val_compare_nb.py"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [],
}

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}

NB["cells"].append(md("""# Val 지표 정리: Sec.6 default vs Optuna+Sec.8 tuned (동일 Val 윈도우)

**목적:** `notebooks/clean/` 메인 노트북과 동일 분할에서 **튜닝 전(Sec.6 default)** vs **튜닝 후(`{BEST_MODEL_NAME}_tuned.pt`)** 의 **Val** 지표(F2, F1, P/R, ROC, PR)를 한 표로 비교합니다.

**전제 (중요):**
- 각 실험(VIF / all-sensors)마다 **그 실행에서 저장한** `models/` 폴더를 지정합니다 (`scaler.pkl`, `final_features.pkl`, `*_default.pt`, `*_tuned.pt` 포함).
- 메인 노트북 **Section 7 마지막**에 출력된 `BEST_PARAMS` 전체를 아래 `PIPELINES[...]['BEST_PARAMS']`에 **복사**합니다 (`study.best_params`와 동일).
- `BEST_MODEL_NAME`도 메인 로그와 동일해야 합니다.

**실행:** Kaggle 또는 로컬 GPU 권장. 데이터 `sensor.csv` 경로는 자동 탐색합니다.

---
"""))

NB["cells"].append(code(r"""
# ══════════════════════════════════════════════════════════════════════════
# CONFIG — 메인 노트북 로그에서 복사해 채우세요
# ══════════════════════════════════════════════════════════════════════════
import os, pickle, warnings
warnings.filterwarnings('ignore')

# 예시 키: 'vif' / 'all' — 표에 쓰일 이름만 구분용
PIPELINES = {
    'vif': {
        'MODEL_DIR': '/kaggle/working/models',   # 해당 실험 Output 경로로 변경
        'BEST_MODEL_NAME': 'CNN1D-AE',         # 예: 'LSTM-AE', 'CNN1D-AE', 'Transformer-AE'
        'BEST_PARAMS': {
            # 아래는 예시 — 실제 study.best_params + setdefault('n_layers',2) 결과로 교체
            'hidden': 64, 'latent': 8, 'n_layers': 2, 'dropout': 0.35,
            'lr': 0.001, 'weight_decay': 1e-5, 'batch': 64,
            'optimizer': 'AdamW', 'threshold_pct': 93, 'window_size': 50,
        },
    },
    'all_sensors': {
        'MODEL_DIR': '/kaggle/working/models_all',
        'BEST_MODEL_NAME': 'LSTM-AE',
        'BEST_PARAMS': {
            'hidden': 128, 'latent': 64, 'n_layers': 2, 'dropout': 0.15,
            'lr': 0.001, 'weight_decay': 1e-5, 'batch': 128,
            'optimizer': 'AdamW', 'threshold_pct': 95, 'window_size': 50,
        },
    },
}

# Sec.6 default와 동일해야 함 (메인 노트북 Section 0)
WINDOW_SIZE_DEFAULT = 50
STEP_SIZE = 10
THRESHOLD_PCT_DEFAULT = 95
SPLIT_TRAIN_FRAC, SPLIT_VAL_FRAC = 0.55, 0.15
SEED = 42

DEFAULT_HIDDEN = 64
DEFAULT_LATENT = 32
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.1

# 비활성화할 파이프라인은 None 또는 주석 처리
ACTIVE = ['vif', 'all_sensors']   # 예: ['vif'] 만 돌리기
"""))

NB["cells"].append(code(r"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ── 데이터 경로 (메인과 동일) ─────────────────────────────────────────────
DATA_PATH = None
for base in ['/kaggle/input', '.', os.path.dirname(os.getcwd())]:
    bp = base if os.path.isdir(base) else None
    if bp is None:
        continue
    for root, _, files in os.walk(bp):
        if 'sensor.csv' in files:
            DATA_PATH = os.path.join(root, 'sensor.csv')
            break
    if DATA_PATH:
        break
if DATA_PATH is None:
    raise FileNotFoundError('sensor.csv 를 찾을 수 없습니다.')
print('DATA_PATH:', DATA_PATH)

# ── 모델 클래스 (메인 Sec.5와 동일) ─────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, layers=2, dropout=0.1, **kw):
        super().__init__()
        drop = dropout if layers > 1 else 0
        self.enc = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=drop)
        self.fc_e = nn.Linear(hidden, latent)
        self.fc_d = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.fc_e(h[-1])
        di = self.fc_d(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.dec(di)
        return self.out(dec)


class CNN1DAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, dropout=0.1, **kw):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_feat, hidden, 7, padding=3), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_e = nn.Linear(hidden * 2, latent)
        self.fc_d = nn.Linear(latent, hidden * 2)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.ConvTranspose1d(hidden, n_feat, 7, padding=3)
        )

    def forward(self, x):
        W = x.size(1)
        z = self.fc_e(self.enc(x.permute(0, 2, 1)).squeeze(-1))
        di = self.fc_d(z).unsqueeze(-1).repeat(1, 1, W)
        return self.dec(di).permute(0, 2, 1)


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, nhead=4, layers=2, dropout=0.1, **kw):
        super().__init__()
        nhead = max(h for h in [1, 2, 4, 8] if hidden % h == 0 and h <= nhead)
        self.proj = nn.Linear(n_feat, hidden)
        enc_l = nn.TransformerEncoderLayer(hidden, nhead, hidden * 4, dropout, batch_first=True)
        self.tenc = nn.TransformerEncoder(enc_l, layers)
        self.fc_e = nn.Linear(hidden, latent)
        self.fc_d = nn.Linear(latent, hidden)
        dec_l = nn.TransformerDecoderLayer(hidden, nhead, hidden * 4, dropout, batch_first=True)
        self.tdec = nn.TransformerDecoder(dec_l, layers)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        p = self.proj(x)
        m = self.tenc(p)
        z = self.fc_e(m.mean(1))
        d = self.fc_d(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.out(self.tdec(d, m))


DEEP_MODEL_CLASSES = {
    'LSTM-AE': LSTMAutoencoder,
    'CNN1D-AE': CNN1DAutoencoder,
    'Transformer-AE': TransformerAutoencoder,
}


def make_windows(data, labels, win, step):
    X, y = [], []
    for s in range(0, len(data) - win, step):
        X.append(data[s:s + win])
        y.append(int(labels[s:s + win].max()))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def get_errors(model, X_tensor, batch=512):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch):
            b = X_tensor[i:i + batch]
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                recon = model(b)
            errs.extend(((b - recon) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.array(errs)


def evaluate_metrics(errors, y_true, pct):
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    thr = np.percentile(errors, pct)
    y_pred = (errors > thr).astype(int)
    out = {
        'threshold': thr,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_true, errors),
    }
    try:
        out['roc_auc'] = roc_auc_score(y_true, errors)
    except ValueError:
        out['roc_auc'] = float('nan')
    return out


def build_window_cache(train_scaled, val_scaled, test_scaled, labels_train, labels_val, labels_test):
    wc = {}
    for ws in [30, 50, 100, 150]:
        Xt, yt = make_windows(train_scaled, labels_train, ws, STEP_SIZE)
        Xv, yv = make_windows(val_scaled, labels_val, ws, STEP_SIZE)
        Xte, yte = make_windows(test_scaled, labels_test, ws, STEP_SIZE)
        _ni = np.where(yt == 0)[0]
        wc[ws] = {
            'X_train': torch.FloatTensor(Xt[_ni]),
            'X_val': torch.FloatTensor(Xv).to(device),
            'y_val': yv,
            'X_test': torch.FloatTensor(Xte),
            'y_test': yte,
        }
    return wc


def load_pipeline(model_dir):
    with open(os.path.join(model_dir, 'final_features.pkl'), 'rb') as f:
        FINAL_FEATURES = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw.columns = df_raw.columns.str.strip()
    ts_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
    if ts_col:
        df_raw.rename(columns={ts_col: 'timestamp'}, inplace=True)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw = df_raw.set_index('timestamp').sort_index()

    for c in FINAL_FEATURES:
        if c not in df_raw.columns:
            raise KeyError(f'Column {c} not in CSV — scaler/features와 데이터가 맞는지 확인하세요.')

    df_proc = df_raw[FINAL_FEATURES + ['machine_status']].copy().reset_index(drop=True)
    label_map = {'NORMAL': 0, 'BROKEN': 1, 'RECOVERING': 1}
    df_proc['label'] = df_proc['machine_status'].map(label_map).fillna(0).astype(int)

    n = len(df_proc)
    i_tr = int(n * SPLIT_TRAIN_FRAC)
    i_val = int(n * (SPLIT_TRAIN_FRAC + SPLIT_VAL_FRAC))
    train_df = df_proc.iloc[:i_tr].copy()
    val_df = df_proc.iloc[i_tr:i_val].copy()
    test_df = df_proc.iloc[i_val:].copy()

    def _impute(df_part):
        df_part = df_part.copy()
        df_part[FINAL_FEATURES] = df_part[FINAL_FEATURES].ffill().fillna(0.0)
        return df_part

    train_df, val_df, test_df = _impute(train_df), _impute(val_df), _impute(test_df)

    def _safe(X):
        return np.nan_to_num(scaler.transform(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    train_scaled = _safe(train_df[FINAL_FEATURES])
    val_scaled = _safe(val_df[FINAL_FEATURES])
    test_scaled = _safe(test_df[FINAL_FEATURES])
    labels_train = train_df['label'].values
    labels_val = val_df['label'].values
    labels_test = test_df['label'].values

    WINDOW_CACHE = build_window_cache(
        train_scaled, val_scaled, test_scaled,
        labels_train, labels_val, labels_test,
    )
    N_FEATURES = len(FINAL_FEATURES)
    return {
        'FINAL_FEATURES': FINAL_FEATURES,
        'N_FEATURES': N_FEATURES,
        'WINDOW_CACHE': WINDOW_CACHE,
        'label_summary': {
            'train_anom': float(train_df['label'].mean()),
            'val_anom': float(val_df['label'].mean()),
            'test_anom': float(test_df['label'].mean()),
        },
    }


def load_state(model, path):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def eval_one_row(tag, model_name, metrics):
    return {
        'pipeline': tag,
        'variant': model_name,
        'F2': metrics['f2'], 'F1': metrics['f1'],
        'Precision': metrics['precision'], 'Recall': metrics['recall'],
        'ROC-AUC': metrics['roc_auc'], 'PR-AUC': metrics['pr_auc'],
        'thr_pct': metrics.get('thr_pct', np.nan),
        'window': metrics.get('window', np.nan),
    }


def run_pipeline(tag, cfg):
    md = cfg['MODEL_DIR']
    name = cfg['BEST_MODEL_NAME']
    p = cfg['BEST_PARAMS'].copy()
    p.setdefault('n_layers', 2)

    ctx = load_pipeline(md)
    WC = ctx['WINDOW_CACHE']
    n_feat = ctx['N_FEATURES']
    print(f"\n=== [{tag}] N_FEATURES={n_feat} | anomaly% train/val/test: "
          f"{ctx['label_summary']['train_anom']*100:.2f} / "
          f"{ctx['label_summary']['val_anom']*100:.2f} / "
          f"{ctx['label_summary']['test_anom']*100:.2f}")

    rows = []

    # --- Sec.6 default: default 가중치, window=50, threshold_pct=THRESHOLD_PCT_DEFAULT ---
    path_def = os.path.join(md, f'{name}_default.pt')
    if not os.path.isfile(path_def):
        raise FileNotFoundError(path_def)
    ModelCls = DEEP_MODEL_CLASSES[name]
    m_def = ModelCls(
        n_feat, hidden=DEFAULT_HIDDEN, latent=DEFAULT_LATENT,
        layers=DEFAULT_LAYERS, dropout=DEFAULT_DROPOUT, nhead=4,
    ).to(device)
    load_state(m_def, path_def)
    cdef = WC[WINDOW_SIZE_DEFAULT]
    err_def = get_errors(m_def, cdef['X_val'])
    met_def = evaluate_metrics(err_def, cdef['y_val'], THRESHOLD_PCT_DEFAULT)
    met_def['thr_pct'] = THRESHOLD_PCT_DEFAULT
    met_def['window'] = WINDOW_SIZE_DEFAULT
    rows.append(eval_one_row(tag, 'Sec.6 default (Val)', met_def))
    print(f"  [default] Val F2={met_def['f2']:.4f}  F1={met_def['f1']:.4f}  "
          f"P={met_def['precision']:.4f}  R={met_def['recall']:.4f}  "
          f"ROC={met_def['roc_auc']:.4f}  PR={met_def['pr_auc']:.4f}")

    # --- Tuned: Sec.8 final weights, BEST_PARAMS window + threshold_pct ---
    path_tuned = os.path.join(md, f'{name}_tuned.pt')
    if not os.path.isfile(path_tuned):
        print(f"  WARNING: {path_tuned} 없음 — Optuna만 돌았다면 Sec.8까지 실행 후 다시 시도.")
    else:
        ws = int(p['window_size'])
        nl = 2 if name == 'CNN1D-AE' else int(p['n_layers'])
        m_t = ModelCls(
            n_feat, hidden=p['hidden'], latent=p['latent'],
            layers=nl, dropout=p['dropout'], nhead=4,
        ).to(device)
        load_state(m_t, path_tuned)
        ct = WC[ws]
        err_t = get_errors(m_t, ct['X_val'])
        met_t = evaluate_metrics(err_t, ct['y_val'], int(p['threshold_pct']))
        met_t['thr_pct'] = int(p['threshold_pct'])
        met_t['window'] = ws
        rows.append(eval_one_row(tag, 'Tuned (Val, Sec.8 weights)', met_t))
        print(f"  [tuned]   Val F2={met_t['f2']:.4f}  F1={met_t['f1']:.4f}  "
              f"P={met_t['precision']:.4f}  R={met_t['recall']:.4f}  "
              f"ROC={met_t['roc_auc']:.4f}  PR={met_t['pr_auc']:.4f}  "
              f"(ws={ws}, pct={p['threshold_pct']})")

    return rows


all_rows = []
for k in ACTIVE:
    if k not in PIPELINES:
        continue
    all_rows.extend(run_pipeline(k, PIPELINES[k]))

summary = pd.DataFrame(all_rows)
print("\n\n========== Val 비교 (동일 데이터·동일 scaler·메인 분할) ==========\n")
print(summary.to_string(index=False))
"""))

NB["cells"][-1]["id"] = "main"

out = REPO / "notebooks" / "val_compare" / "val_compare.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out.relative_to(REPO))
