"""
app.py
Gradio 기반 Pump Sensor Anomaly Detection 로컬 대시보드
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

load_dotenv()

# 한국어 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
DATA_DIR       = os.path.join(BASE_DIR, 'data', 'raw')
FIGURES_DIR    = os.path.join(BASE_DIR, 'outputs', 'figures')
REPORTS_DIR    = os.path.join(BASE_DIR, 'outputs', 'reports')

for d in [FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
#  모델 정의
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


MODEL_CLASSES = {'LSTM-AE': LSTMAutoencoder, 'CNN1D-AE': CNN1DAutoencoder}
BEST_PARAMS = {
    'LSTM-AE':  {'hidden': 128, 'latent': 64,  'layers': 2, 'dropout': 0.15},
    'CNN1D-AE': {'hidden': 64,  'latent': 8,   'layers': 2, 'dropout': 0.35},
}
WINDOW_CONFIGS = {
    'LSTM-AE':  {'window_size': 30,  'threshold_pct': 93},
    'CNN1D-AE': {'window_size': 50,  'threshold_pct': 93},
}


# ══════════════════════════════════════════════════════════════════
#  모델 로더
# ══════════════════════════════════════════════════════════════════
_model_cache: Dict = {}

def load_model(model_name: str, model_type: str = 'tuned'):
    cache_key = f'{model_name}_{model_type}'
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    pt_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_{model_type}.pt')
    if not os.path.exists(pt_path):
        # fallback: optuna_best → default
        for t in ['optuna_best', 'default']:
            alt = os.path.join(CHECKPOINT_DIR, f'{model_name}_{t}.pt')
            if os.path.exists(alt):
                pt_path = alt
                break
        else:
            raise FileNotFoundError(
                f'{model_name} 모델 파일을 찾을 수 없습니다.\n'
                f'경로: {CHECKPOINT_DIR}\n'
                f'Kaggle Output에서 .pt 파일을 다운로드하세요.'
            )

    feat_path = os.path.join(CHECKPOINT_DIR, 'final_features.pkl')
    with open(feat_path, 'rb') as f:
        features = pickle.load(f)
    n_feat = len(features)

    params = BEST_PARAMS.get(model_name, {})
    model = MODEL_CLASSES[model_name](n_feat, **params).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    _model_cache[cache_key] = (model, features)
    return model, features


def load_scaler():
    path = os.path.join(CHECKPOINT_DIR, 'scaler.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
#  추론
# ══════════════════════════════════════════════════════════════════
def make_windows(data: np.ndarray, win: int, step: int) -> np.ndarray:
    X = [data[s:s + win] for s in range(0, len(data) - win, step)]
    return np.array(X, dtype=np.float32)


def get_errors(model, X_tensor, batch=512) -> np.ndarray:
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch):
            b = X_tensor[i:i + batch]
            recon = model(b)
            errs.extend(((b - recon) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.nan_to_num(np.array(errs), nan=0.0, posinf=0.0, neginf=0.0)


def detect(errors: np.ndarray, pct: int) -> Tuple[np.ndarray, float]:
    thr = np.percentile(errors, pct)
    return (errors > thr).astype(int), thr


def extract_segments(errors: np.ndarray, y_pred: np.ndarray, thr: float) -> List[Dict]:
    segs, in_seg = [], False
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and not in_seg:
            start, in_seg = i, True
        elif y_pred[i] == 0 and in_seg:
            _add_seg(segs, errors, start, i, thr)
            in_seg = False
    if in_seg:
        _add_seg(segs, errors, start, len(y_pred), thr)
    return segs


def _add_seg(segs, errors, start, end, thr):
    seg_errs = errors[start:end]
    max_err = float(seg_errs.max())
    sev = 'HIGH' if max_err > thr * 3 else 'MEDIUM' if max_err > thr * 1.5 else 'LOW'
    segs.append({
        'start': start, 'end': end,
        'duration': end - start,
        'max_err': round(max_err, 5),
        'mean_err': round(float(seg_errs.mean()), 5),
        'severity': sev,
    })


def run_inference(df: pd.DataFrame, model_name: str, model_type: str, threshold_pct: int):
    # 업로드 데이터에 machine_status가 있으면 이진 라벨 생성
    if 'machine_status' in df.columns and 'label' not in df.columns:
        df = df.copy()
        df['label'] = (df['machine_status'] != 'NORMAL').astype(int)

    scaler = load_scaler()
    model, features = load_model(model_name, model_type)
    win = WINDOW_CONFIGS[model_name]['window_size']
    step = 10

    data_df = df[features].ffill().bfill()
    data_scaled = scaler.transform(data_df).astype(np.float32)
    X = make_windows(data_scaled, win, step)
    if len(X) == 0:
        raise ValueError(f'데이터가 너무 짧습니다. 최소 {win + step}행 필요.')

    X_t = torch.FloatTensor(X).to(device)
    errors = get_errors(model, X_t)
    y_pred, thr = detect(errors, threshold_pct)
    segments = extract_segments(errors, y_pred, thr)

    result = {
        'errors':        errors,
        'y_pred':        y_pred,
        'threshold':     thr,
        'segments':      segments,
        'n_windows':     len(errors),
        'n_anomaly':     int(y_pred.sum()),
        'anomaly_ratio': round(y_pred.mean() * 100, 2),
    }

    if 'label' in df.columns:
        labels = df['label'].values
        y_true = np.array([
            int(labels[s:s+win].max())
            for s in range(0, len(labels) - win, step)
        ], dtype=np.int32)
        y_true = y_true[:len(y_pred)]

        metrics = {'f1': 'N/A', 'roc_auc': 'N/A', 'pr_auc': 'N/A'}
        if len(y_true) > 0:
            metrics['f1'] = round(f1_score(y_true, y_pred, zero_division=0), 4)
            if np.unique(y_true).size > 1:
                metrics['roc_auc'] = round(roc_auc_score(y_true, errors), 4)
                metrics['pr_auc'] = round(average_precision_score(y_true, errors), 4)
        result['metrics'] = metrics

    return result


# ══════════════════════════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════════════════════════
def plot_result(result: Dict, model_name: str) -> plt.Figure:
    errors   = result['errors']
    y_pred   = result['y_pred']
    thr      = result['threshold']
    segments = result['segments']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Anomaly Detection — {model_name}', fontsize=14, fontweight='bold')

    # ── 위: 재구성 오차 ──────────────────────────────────────────
    ax1 = axes[0]
    idx = np.arange(len(errors))
    ax1.plot(idx, errors, color='#6366F1', linewidth=0.8, label='재구성 오차')
    ax1.axhline(thr, color='#EF4444', linestyle='--', linewidth=1.2, label=f'임계값 ({thr:.5f})')
    for s in segments:
        color = '#EF4444' if s['severity'] == 'HIGH' else '#F59E0B' if s['severity'] == 'MEDIUM' else '#10B981'
        ax1.axvspan(s['start'], s['end'], alpha=0.25, color=color)
    ax1.set_ylabel('재구성 오차 (MSE)')
    ax1.set_title(f'재구성 오차 | 이상 구간: {len(segments)}개 | 이상 비율: {result["anomaly_ratio"]}%')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 아래: 이상 탐지 결과 ─────────────────────────────────────
    ax2 = axes[1]
    colors_pred = ['#EF4444' if p == 1 else '#6EE7B7' for p in y_pred]
    ax2.bar(idx, y_pred, color=colors_pred, width=1.0, alpha=0.7)
    ax2.set_ylabel('이상 여부')
    ax2.set_xlabel('윈도우 인덱스')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['정상', '이상'])
    ax2.set_title('이상 탐지 결과')
    patches = [
        mpatches.Patch(color='#EF4444', label='HIGH'),
        mpatches.Patch(color='#F59E0B', label='MEDIUM'),
        mpatches.Patch(color='#10B981', label='LOW'),
    ]
    ax2.legend(handles=patches, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def build_segment_table(segments: List[Dict]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame({'메시지': ['탐지된 이상 구간 없음']})
    rows = [{
        '구간': f'{s["start"]}~{s["end"]}',
        '지속 (윈도우)': s['duration'],
        '심각도': s['severity'],
        '최대 오차': s['max_err'],
        '평균 오차': s['mean_err'],
    } for s in segments]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  Gradio 이벤트 핸들러
# ══════════════════════════════════════════════════════════════════
def run_analysis(
    file,
    model_choice: str,
    model_type: str,
    threshold_pct: int,
    generate_llm: bool,
):
    # ── 데이터 로드 ──────────────────────────────────────────────
    try:
        if file is None:
            # 샘플 데이터 생성 (데모용)
            sample_path = os.path.join(DATA_DIR, 'sensor.csv')
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path, nrows=5000, index_col=0)
            else:
                return None, pd.DataFrame({'안내': ['CSV 파일을 업로드하거나 data/raw/sensor.csv를 배치하세요.']}), ''
        else:
            df = pd.read_csv(file.name, index_col=0)
    except Exception as e:
        return None, pd.DataFrame({'에러': [str(e)]}), f'데이터 로드 오류: {e}'

    # ── 추론 ─────────────────────────────────────────────────────
    try:
        result = run_inference(df, model_choice, model_type, threshold_pct)
    except FileNotFoundError as e:
        return None, pd.DataFrame({'에러': [str(e)]}), str(e)
    except Exception as e:
        return None, pd.DataFrame({'에러': [str(e)]}), f'추론 오류: {e}'

    # ── 시각화 ───────────────────────────────────────────────────
    fig = plot_result(result, model_choice)

    # ── 세그먼트 테이블 ──────────────────────────────────────────
    seg_table = build_segment_table(result['segments'])

    # ── LLM 리포트 ───────────────────────────────────────────────
    report_text = ''
    if generate_llm:
        try:
            from rag_pipeline import generate_report
            metrics = result.get('metrics', {'f1': 'N/A', 'roc_auc': 'N/A', 'pr_auc': 'N/A'})
            report_text = generate_report(
                result['segments'],
                metrics,
                f'{model_choice} ({model_type})'
            )
            # 리포트 저장
            report_path = os.path.join(REPORTS_DIR, 'latest_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        except Exception as e:
            report_text = f'리포트 생성 오류: {e}'

    summary = (
        f'**분석 완료** | 모델: {model_choice} ({model_type}) | '
        f'총 윈도우: {result["n_windows"]} | '
        f'이상 윈도우: {result["n_anomaly"]} ({result["anomaly_ratio"]}%) | '
        f'이상 구간: {len(result["segments"])}개'
    )

    return fig, seg_table, f'{summary}\n\n{report_text}'


# ══════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════
def create_ui():
    with gr.Blocks(
        title='Pump Sensor Anomaly Detection'
    ) as demo:

        gr.Markdown("""
# 🔬 Pump Sensor Anomaly Detection
**LSTM/CNN Autoencoder + Optuna 튜닝 + RAG 기반 LLM 진단 리포트**
        """)

        with gr.Row():
            # ── 사이드바 ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown('### ⚙️ 설정')

                file_input = gr.File(
                    label='센서 CSV 업로드 (없으면 기본 데이터 사용)',
                    file_types=['.csv']
                )

                model_choice = gr.Dropdown(
                    label='모델 선택',
                    choices=['CNN1D-AE', 'LSTM-AE'],
                    value='CNN1D-AE'
                )

                model_type = gr.Dropdown(
                    label='모델 타입',
                    choices=['tuned', 'optuna_best', 'default'],
                    value='tuned'
                )

                threshold_pct = gr.Slider(
                    label='이상 탐지 임계값 (percentile)',
                    minimum=80,
                    maximum=99,
                    value=93,
                    step=1
                )

                generate_llm = gr.Checkbox(
                    label='LLM 진단 리포트 생성 (Gemini API)',
                    value=True
                )

                run_btn = gr.Button('🚀 분석 실행', variant='primary', size='lg')

            # ── 메인 패널 ────────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown('### 📊 탐지 결과')
                plot_output = gr.Plot(label='재구성 오차 & 이상 구간')

                gr.Markdown('### 📋 이상 구간 상세')
                table_output = gr.Dataframe(label='이상 구간 목록')

                gr.Markdown('### 📝 RAG 기반 진단 리포트')
                report_output = gr.Markdown(label='진단 리포트')

        run_btn.click(
            fn=run_analysis,
            inputs=[file_input, model_choice, model_type, threshold_pct, generate_llm],
            outputs=[plot_output, table_output, report_output]
        )

        gr.Markdown("""
---
**사용 방법:**
1. CSV 파일 업로드 (sensor_00~51 컬럼 포함) 또는 기본 데이터 사용
2. 모델 선택 → 임계값 조절 → 분석 실행
3. LLM 리포트는 `.env`에 `GEMINI_API_KEY` 설정 시 활성화

**모델 파일 경로:** `models/checkpoints/` (.pt, .pkl 파일 필요)
        """)

    return demo


# ══════════════════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'Checkpoint: {CHECKPOINT_DIR}')

    # RAG 초기화
    try:
        from rag_pipeline import get_db
        get_db()
        print('[RAG] 초기화 완료')
    except Exception as e:
        print(f'[RAG] 초기화 스킵: {e}')

    demo = create_ui()
    demo.launch(
        share=True,
        server_name='0.0.0.0',
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft()
    )
