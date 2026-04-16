"""
inference.py
Kaggle에서 다운로드한 모델로 로컬 이상 탐지 추론
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# ── 디바이스 ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
#  모델 정의 (Kaggle 노트북과 동일)
# ══════════════════════════════════════════════════════════════════

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, layers=2, dropout=0.1, **kw):
        super().__init__()
        drop = dropout if layers > 1 else 0
        self.enc  = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=drop)
        self.fc_e = nn.Linear(hidden, latent)
        self.fc_d = nn.Linear(latent, hidden)
        self.dec  = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.out  = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z  = self.fc_e(h[-1])
        di = self.fc_d(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.dec(di)
        return self.out(dec)


class CNN1DAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, dropout=0.1, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            nn.Conv1d(n_feat, hidden, 7, padding=3), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden*2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_e = nn.Linear(hidden*2, latent)
        self.fc_d = nn.Linear(latent, hidden*2)
        self.dec  = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.ConvTranspose1d(hidden, n_feat, 7, padding=3)
        )

    def forward(self, x):
        W = x.size(1)
        z  = self.fc_e(self.enc(x.permute(0,2,1)).squeeze(-1))
        di = self.fc_d(z).unsqueeze(-1).repeat(1,1,W)
        return self.dec(di).permute(0,2,1)


MODEL_CLASSES = {
    'LSTM-AE':  LSTMAutoencoder,
    'CNN1D-AE': CNN1DAutoencoder,
}

# Best Params (Kaggle 결과)
BEST_PARAMS = {
    'LSTM-AE': {
        'hidden': 128, 'latent': 64, 'layers': 2, 'dropout': 0.15
    },
    'CNN1D-AE': {
        'hidden': 64, 'latent': 8, 'layers': 2, 'dropout': 0.35
    },
}


# ══════════════════════════════════════════════════════════════════
#  모델 로더
# ══════════════════════════════════════════════════════════════════

def load_model(model_name: str, model_type: str, checkpoint_dir: str):
    """
    model_name: 'LSTM-AE' | 'CNN1D-AE'
    model_type: 'tuned' | 'optuna_best' | 'default'
    """
    with open(os.path.join(checkpoint_dir, 'final_features.pkl'), 'rb') as f:
        features = pickle.load(f)
    n_feat = len(features)

    params = BEST_PARAMS.get(model_name, {})
    ModelCls = MODEL_CLASSES[model_name]
    model = ModelCls(n_feat, **params).to(device)

    pt_path = os.path.join(checkpoint_dir, f'{model_name}_{model_type}.pt')
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    return model, features


def load_scaler(checkpoint_dir: str):
    with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'rb') as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
#  전처리
# ══════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, features: list, scaler, window_size: int = 50, step_size: int = 10):
    """
    센서 데이터프레임 → 슬라이딩 윈도우 텐서
    """
    df_proc = df[features].copy().ffill().bfill()
    data_scaled = scaler.transform(df_proc).astype(np.float32)

    X = []
    for s in range(0, len(data_scaled) - window_size, step_size):
        X.append(data_scaled[s:s+window_size])
    X = np.array(X, dtype=np.float32)
    return torch.FloatTensor(X).to(device)


# ══════════════════════════════════════════════════════════════════
#  추론
# ══════════════════════════════════════════════════════════════════

def get_recon_errors(model, X_tensor, batch=512):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch):
            b = X_tensor[i:i+batch]
            recon = model(b)
            errs.extend(((b - recon)**2).mean(dim=(1,2)).cpu().numpy())
    return np.array(errs)


def detect_anomalies(errors: np.ndarray, threshold_pct: int = 93):
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = np.percentile(errors, threshold_pct)
    y_pred = (errors > threshold).astype(int)
    return y_pred, threshold


def extract_segments(errors: np.ndarray, y_pred: np.ndarray, threshold: float):
    """이상 구간별 통계 추출"""
    segments = []
    in_seg = False
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and not in_seg:
            start = i
            in_seg = True
        elif y_pred[i] == 0 and in_seg:
            seg_errors = errors[start:i]
            max_err = float(seg_errors.max())
            severity = 'HIGH' if max_err > threshold * 3 else 'MEDIUM' if max_err > threshold * 1.5 else 'LOW'
            segments.append({
                'start': start,
                'end': i,
                'duration': i - start,
                'max_err': round(max_err, 5),
                'mean_err': round(float(seg_errors.mean()), 5),
                'severity': severity,
            })
            in_seg = False
    if in_seg:
        seg_errors = errors[start:]
        max_err = float(seg_errors.max())
        severity = 'HIGH' if max_err > threshold * 3 else 'MEDIUM' if max_err > threshold * 1.5 else 'LOW'
        segments.append({
            'start': start,
            'end': len(y_pred),
            'duration': len(y_pred) - start,
            'max_err': round(max_err, 5),
            'mean_err': round(float(seg_errors.mean()), 5),
            'severity': severity,
        })
    return segments


def run_inference(
    df: pd.DataFrame,
    model_name: str = 'CNN1D-AE',
    model_type: str = 'tuned',
    checkpoint_dir: str = './models/checkpoints',
    window_size: int = 50,
    step_size: int = 10,
    threshold_pct: int = 93,
):
    """
    메인 추론 함수.
    Returns: errors, y_pred, threshold, segments, metrics(y_true 있을 때)
    """
    # 업로드 데이터에 machine_status가 있으면 이진 라벨 생성
    if 'machine_status' in df.columns and 'label' not in df.columns:
        df = df.copy()
        df['label'] = (df['machine_status'] != 'NORMAL').astype(int)

    scaler   = load_scaler(checkpoint_dir)
    model, features = load_model(model_name, model_type, checkpoint_dir)

    X_tensor = preprocess(df, features, scaler, window_size, step_size)
    errors   = get_recon_errors(model, X_tensor)
    y_pred, threshold = detect_anomalies(errors, threshold_pct)
    segments = extract_segments(errors, y_pred, threshold)

    result = {
        'errors':    errors,
        'y_pred':    y_pred,
        'threshold': threshold,
        'segments':  segments,
        'n_windows': len(errors),
        'n_anomaly': int(y_pred.sum()),
        'anomaly_ratio': round(y_pred.mean() * 100, 2),
    }

    # y_true 있으면 지표 계산
    if 'label' in df.columns:
        labels = df['label'].values
        y_true = np.array([
            int(labels[s:s+window_size].max())
            for s in range(0, len(labels) - window_size, step_size)
        ], dtype=np.int32)
        y_true = y_true[:len(y_pred)]
        metrics = {
            'f1': round(f1_score(y_true, y_pred, zero_division=0), 4) if len(y_true) > 0 else 'N/A',
            'roc_auc': 'N/A',
            'pr_auc': 'N/A',
        }
        if len(y_true) > 0 and np.unique(y_true).size > 1:
            metrics['roc_auc'] = round(roc_auc_score(y_true, errors), 4)
            metrics['pr_auc'] = round(average_precision_score(y_true, errors), 4)
        result['metrics'] = metrics

    return result
