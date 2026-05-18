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
from typing import Dict, List, Optional

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from schema_validation import validate_feature_columns
from serving_config import load_serving_config


def supervised_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray,
) -> Dict:
    """
    라벨이 있을 때 윈도우 단위 혼동·F1·AUC(가능할 때만).
    라벨: 윈도우 구간 내 max, 예측: 재구성 오차 백분위 임계 초과.
    """
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    m: Dict = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'n_pos_windows': n_pos,
        'n_neg_windows': n_neg,
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc': 'N/A',
        'pr_auc': 'N/A',
    }
    notes: List[str] = [
        f'윈도우 혼동 TN={tn}, FP={fp}, FN={fn}, TP={tp} '
        f'(이상 라벨 윈도우 {n_pos}개 / 정상 {n_neg}개).',
    ]
    if n_pos > 0 and n_neg > 0:
        m['roc_auc'] = round(roc_auc_score(y_true, errors), 4)
        m['pr_auc'] = round(average_precision_score(y_true, errors), 4)
        m['precision'] = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
        m['recall'] = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    else:
        if n_pos == 0:
            notes.append(
                '이 구간에서는 라벨 기준「이상」윈도우가 없어 ROC-AUC·PR-AUC는 계산하지 않았습니다.'
            )
        else:
            notes.append(
                '라벨이 모두 이상이라 AUC는 참고용이며, 임계값·운전 구간을 함께 검토하세요.'
            )

    m['metrics_note'] = ' '.join(notes)
    return m


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

    cfg = load_serving_config(checkpoint_dir)
    params = cfg['best_params'].get(model_name, {})
    ModelCls = MODEL_CLASSES[model_name]
    model = ModelCls(n_feat, **params).to(device)

    pt_path = os.path.join(checkpoint_dir, f'{model_name}_{model_type}.pt')
    if not os.path.isfile(pt_path):
        for t in ('optuna_best', 'default'):
            alt = os.path.join(checkpoint_dir, f'{model_name}_{t}.pt')
            if os.path.isfile(alt):
                pt_path = alt
                break
        else:
            raise FileNotFoundError(
                f'{model_name}용 .pt를 찾을 수 없습니다: {checkpoint_dir} '
                f'({model_type} / optuna_best / default)'
            )
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
    센서 데이터프레임 → 슬라이딩 윈도우 텐서.
    학습 노트북과 동일: ffill만(실시간 가정, bfill 없음) → fillna(0) → scaler.transform → nan_to_num
    """
    df_proc = df[features].copy().ffill().fillna(0.0)
    data_scaled = np.nan_to_num(
        scaler.transform(df_proc), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

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


def get_per_feature_mse(model, X_tensor, batch=512):
    """
    윈도우별 피처 차원 평균 재구성 오차 (시간축 평균).
    반환: (n_windows, n_features) — 스케일된 입력 기준 MSE.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch):
            b = X_tensor[i:i + batch]
            recon = model(b)
            mse_f = ((b - recon) ** 2).mean(dim=1).cpu().numpy()
            chunks.append(mse_f)
    out = np.vstack(chunks)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def aggregate_top3_sensors(per_feat_mse: np.ndarray, y_pred: np.ndarray, feature_names: list):
    """이상 윈도우에서만 피처별 오차를 평균해 기여도 상위 3개 센서."""
    mask = y_pred == 1
    if not np.any(mask):
        return []
    avg = per_feat_mse[mask].mean(axis=0)
    idx = np.argsort(avg)[::-1][:3]
    return [(feature_names[i], float(avg[i])) for i in idx]


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
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    threshold_pct: Optional[int] = None,
):
    """
    비지도 AE 재구성 오차 기반 탐지. 체크포인트의 scaler·피처 목록으로만 transform.
    window_size / step_size / threshold_pct 가 None 이면 checkpoint_dir의
    model_serving_config.json (없으면 코드 기본값)을 사용합니다.
    """
    # 업로드 데이터에 machine_status가 있으면 이진 라벨 생성 (평가용)
    if 'machine_status' in df.columns and 'label' not in df.columns:
        df = df.copy()
        df['label'] = (df['machine_status'] != 'NORMAL').astype(int)

    cfg = load_serving_config(checkpoint_dir)
    wc = cfg['window_configs'].get(model_name)
    if wc is None:
        wc = cfg['window_configs']['CNN1D-AE']
    window_size = wc['window_size'] if window_size is None else window_size
    step_size = cfg['step_size'] if step_size is None else step_size
    threshold_pct = wc['threshold_pct'] if threshold_pct is None else threshold_pct

    scaler   = load_scaler(checkpoint_dir)
    model, features = load_model(model_name, model_type, checkpoint_dir)
    schema_note_message, _ = validate_feature_columns(df, features)

    X_tensor = preprocess(df, features, scaler, window_size, step_size)
    errors   = get_recon_errors(model, X_tensor)
    per_feat = get_per_feature_mse(model, X_tensor)
    y_pred, threshold = detect_anomalies(errors, threshold_pct)
    segments = extract_segments(errors, y_pred, threshold)
    top3 = aggregate_top3_sensors(per_feat, y_pred, features)

    result = {
        'errors':    errors,
        'y_pred':    y_pred,
        'threshold': threshold,
        'segments':  segments,
        'n_windows': len(errors),
        'n_anomaly': int(y_pred.sum()),
        'anomaly_ratio': round(y_pred.mean() * 100, 2),
        'top3_sensors': top3,
        'schema_note': schema_note_message,
    }

    # y_true 있으면 지표 계산
    if 'label' in df.columns:
        labels = df['label'].values
        y_true = np.array([
            int(labels[s:s+window_size].max())
            for s in range(0, len(labels) - window_size, step_size)
        ], dtype=np.int32)
        y_true = y_true[:len(y_pred)]
        if len(y_true) > 0:
            result['metrics'] = supervised_window_metrics(y_true, y_pred, errors)

    return result
