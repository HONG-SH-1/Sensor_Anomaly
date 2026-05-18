"""
app.py
Gradio 기반 Pump Sensor Anomaly Detection 로컬 대시보드
"""

import html
import os
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from inference import run_inference, device

load_dotenv()

# 플롯: 한글 + 라틴·숫자가 섞여도 깨지지 않도록 sans-serif 스택 (웹과 유사)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Malgun Gothic', 'Segoe UI', 'Apple SD Gothic Neo', 'DejaVu Sans', 'Arial', 'sans-serif',
]
plt.rcParams['axes.unicode_minus'] = False

# Gradio UI: 영문·숫자는 Segoe UI 우선, 한글은 맑은 고딕 등으로 폴백 (전역 * 지정은 일부 컴포넌트 깨짐 방지)
APP_UI_CSS = """
.gradio-container {
  font-family: "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container .markdown, .gradio-container table, .gradio-container .dataframe-wrap {
  font-family: "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif !important;
}
/* 상단 설정 바: 컴포넌트 간 간격 */
.app-settings-row {
  flex-wrap: wrap !important;
  align-items: flex-end !important;
  gap: 0.5rem 0.75rem;
}
/* 상단 워크스페이스(사이드바 + CSV·용어) — 한 블록 */
.app-workspace-row {
  width: 100% !important;
  align-items: flex-start !important;
}
.app-input-panel {
  min-width: 0 !important;
  flex: 1 1 0% !important;
}
/* 분석 결과: Row 밖 별도 컬럼 → 풀폭 (라이트 테마) */
.app-analysis-results-root {
  width: 100% !important;
  max-width: 100% !important;
  flex: 1 1 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
  margin-top: 1.25rem !important;
  padding: 1rem 0.25rem 0.5rem 0.25rem !important;
  border-top: 1px solid #e2e8f0 !important;
}
.app-analysis-results-root .plot-container,
.app-analysis-results-root .plot-container > div,
.app-analysis-results-root .js-plotly-plot,
.app-analysis-results-root .plotly-graph-div {
  width: 100% !important;
  max-width: 100% !important;
}
.app-analysis-results-root .dataframe-wrap {
  width: 100% !important;
  max-width: 100% !important;
}
.app-result-section {
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  margin: 0 0 1rem 0 !important;
  padding: 0.85rem 1rem !important;
  border-radius: 12px !important;
  border: 1px solid #e2e8f0 !important;
  background: #ffffff !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
}
/* 좌측 설정 패널 — 라이트 카드 */
.app-sidebar {
  flex: 0 0 min(100%, 340px) !important;
  max-width: 360px !important;
  border-radius: 12px;
  padding: 0.85rem 0.65rem !important;
  border: 1px solid #e2e8f0 !important;
  background: #ffffff !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
}
@media (min-width: 1100px) {
  .app-sidebar { position: sticky; top: 0.75rem; align-self: flex-start; }
}
.app-sidebar label { font-size: 0.88rem !important; }
.app-sidebar .wrap { gap: 0.35rem !important; }
/* 업로드: 큰 드롭존 억제 (한 줄 업로드 + 버튼) */
.app-upload-row .file-preview,
.app-upload-row .upload-container,
.app-upload-row [class*="upload"] {
  min-height: 72px !important;
  max-height: 96px !important;
}
.app-upload-row .upload-box { padding: 0.45rem 0.6rem !important; }
/* 분석 결과: 블록별 구분 (가로 전체) */
.app-analysis-stack {
  width: 100% !important;
  max-width: 100% !important;
}
.app-analysis-stack .plot-container {
  min-height: 420px;
}
.app-section-h {
  margin: 1.25rem 0 0.5rem 0 !important;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid #e2e8f0;
  color: #0f172a !important;
  font-size: 1.05rem !important;
}
.app-analysis-results-root h4 {
  margin: 0.15rem 0 0.5rem 0 !important;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid #e2e8f0;
  color: #0f172a !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
}
.app-analysis-results-root h5 {
  margin: 0.15rem 0 0.4rem 0 !important;
  color: #475569 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
}
.app-empty-hint {
  opacity: 0.95;
  font-size: 0.95rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  border: 1px dashed #fca5a5;
  background: #fef2f2;
  color: #991b1b !important;
}
/* KPI 카드 (gr.HTML) — 라이트 */
.dash-wrap { width: 100%; margin: 0.5rem 0 1rem 0; }
.dash-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.dash-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 0.85rem 1rem;
}
.dash-card-h { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.35rem; }
.dash-card-v { font-size: 1.75rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
.dash-card-v .sub { font-size: 1rem; font-weight: 400; color: #64748b; }
.dash-card-note { font-size: 0.68rem; color: #64748b; margin-top: 0.25rem; }
.dash-mono { font-family: ui-monospace, Consolas, monospace !important; font-size: 1rem !important; color: #0f172a !important; }
.dash-st-ok { color: #15803d !important; }
.dash-st-warn { color: #b45309 !important; }
.dash-st-bad { color: #b91c1c !important; }
.dash-top3 {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1rem 1.1rem;
}
.dash-top3-title { font-size: 0.9rem; font-weight: 600; color: #0f172a; margin-bottom: 0.75rem; }
.dash-bar { margin-bottom: 0.65rem; }
.dash-bar-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #334155; margin-bottom: 0.2rem; }
.dash-bar-label .mono { font-family: ui-monospace, Consolas, monospace; color: #64748b; font-size: 0.72rem; }
.dash-bar-track { height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; }
.dash-bar-fill { height: 100%; border-radius: 99px; transition: width 0.4s ease; }
.dash-muted { color: #64748b; font-size: 0.85rem; margin: 0; }
.dash-err { border: 1px solid #fecaca !important; background: #fef2f2; color: #991b1b; padding: 0.75rem; border-radius: 10px; font-size: 0.9rem; }
"""

# 상단 소개 (짧게 — 상세는 하단 사용 방법)
APP_INTRO_MARKDOWN = """
### Pump Sensor Anomaly Detection
**좌측:** 프리셋 · 모델 · 임계값 · 실행 **우측:** CSV → KPI·차트·Top-3·리포트  
*변경 후에는 서버 재시작 + 브라우저 강력 새로고침(Ctrl+F5).*
"""

APP_GLOSSARY_MARKDOWN = """
#### 체크포인트 프리셋
**어떤 학습 결과 폴더**를 쓸지 고르는 설정입니다. 프리셋마다 `models/checkpoints_…/` 안의 가중치(.pt), 스케일러, 피처 목록(`final_features.pkl`)이 다릅니다. **업로드한 CSV 컬럼**이 해당 프리셋과 맞아야 합니다.

#### 모델 타입
같은 모델 이름(CNN1D-AE 등)이라도 **저장본 종류**가 다릅니다. 화면의 한글 항목은 디스크의 `모델이름_tuned.pt`, `…_optuna_best.pt`, `…_default.pt` 파일과 대응합니다.

#### 재구성 오차 (Reconstruction error)
오토인코더가 입력 윈도우를 복원할 때의 **평균 제곱 오차(MSE)** 입니다. 정상에 가까울수록 작고, 패턴이 학습과 다르면 커질 수 있습니다.

#### 임계값 (percentile)
현재 윈도우들의 오차 분포에서 **상위 (100−p)%** 를 이상으로 자릅니다. 슬라이더가 높을수록(99에 가까울수록) **보수적**(이상 적게 잡힘), 낮을수록 **민감**합니다.

#### 이상 구간
연속된 윈도우가 이상으로 분류된 **구간**입니다. 심각도(HIGH/MEDIUM/LOW)는 임계값 대비 최대 오차 배수로 **상대 등급**입니다.

#### Top-3 의심 센서
**이상으로 분류된 윈도우들만** 모아, 센서(피처)별 **평균 재구성 MSE**가 큰 순서 상위 3개입니다. **물리적 고장 원인 단정은 아니며**, 점검 우선순위 후보입니다.

#### RAG 리포트
규정 텍스트를 검색해 LLM이 문장을 생성합니다. **매뉴얼 원문과 반드시 대조**하세요(근거 점검 경고 참고).

#### 라벨이 있을 때 지표 (Precision 등)
`machine_status`/`label`이 있으면 **윈도우 단위**로 참고 지표를 냅니다. 구간·임계값·라벨 정의에 따라 달라질 수 있습니다.
"""

DASHBOARD_EMPTY_HTML = """
<div class="dash-wrap">
  <p class="dash-muted" style="text-align:center;padding:0.5rem 0;">분석 실행 후 이상 비율·구간·Top-3 기여도가 카드 형태로 표시됩니다.</p>
</div>
"""

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
CHECKPOINT_LEGACY = os.path.join(MODELS_DIR, 'checkpoints')
CHECKPOINT_ALL    = os.path.join(MODELS_DIR, 'checkpoints_all')
CHECKPOINT_VIF    = os.path.join(MODELS_DIR, 'checkpoints_vif')

PRESET_LABELS = {
    'all': '전체 피처 (51센서) — LSTM-AE 권장',
    'vif': 'VIF 선별 (21센서) — CNN1D-AE 권장',
    'legacy': '레거시 — 단일 checkpoints 폴더',
}

# UI 한글 표시 ↔ 로드 시 파일 접미사 (inference `*_tuned.pt` 등)
MODEL_TYPE_CHOICES: List[Tuple[str, str]] = [
    ('튜닝 가중치 (권장)', 'tuned'),
    ('Optuna 탐색 최적', 'optuna_best'),
    ('기본 학습', 'default'),
]
MODEL_TYPE_EN_TO_KO = {internal: disp for disp, internal in MODEL_TYPE_CHOICES}


def _model_type_label(model_type_en: str) -> str:
    return MODEL_TYPE_EN_TO_KO.get(model_type_en, model_type_en)


def _build_checkpoint_presets() -> Dict[str, str]:
    m: Dict[str, str] = {}
    if os.path.isdir(CHECKPOINT_ALL):
        m['all'] = CHECKPOINT_ALL
    if os.path.isdir(CHECKPOINT_VIF):
        m['vif'] = CHECKPOINT_VIF
    if os.path.isdir(CHECKPOINT_LEGACY):
        m['legacy'] = CHECKPOINT_LEGACY
    if not m:
        os.makedirs(CHECKPOINT_ALL, exist_ok=True)
        os.makedirs(CHECKPOINT_VIF, exist_ok=True)
        return {'all': CHECKPOINT_ALL, 'vif': CHECKPOINT_VIF}
    return m


CHECKPOINT_PRESETS = _build_checkpoint_presets()


def _default_preset_key() -> str:
    env = os.environ.get('CHECKPOINT_PRESET', '').strip().lower()
    if env in CHECKPOINT_PRESETS:
        return env
    for k in ('vif', 'all', 'legacy'):
        if k in CHECKPOINT_PRESETS:
            return k
    return next(iter(CHECKPOINT_PRESETS))


def _default_model_for_preset(preset_key: str) -> str:
    if preset_key == 'all':
        return 'LSTM-AE'
    if preset_key == 'vif':
        return 'CNN1D-AE'
    return 'CNN1D-AE'


ACTIVE_PRESET_KEY = _default_preset_key()
CHECKPOINT_DIR = CHECKPOINT_PRESETS[ACTIVE_PRESET_KEY]
DATA_DIR       = os.path.join(BASE_DIR, 'data', 'raw')
SAMPLE_UPLOAD  = os.path.join(BASE_DIR, 'data', 'sample_upload', 'pump_sensor_sample.csv')
FIGURES_DIR    = os.path.join(BASE_DIR, 'outputs', 'figures')
REPORTS_DIR    = os.path.join(BASE_DIR, 'outputs', 'reports')

for d in [FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def set_checkpoint_preset(preset_key: str) -> None:
    """프리셋 변경 시 CHECKPOINT_DIR 갱신."""
    global CHECKPOINT_DIR, ACTIVE_PRESET_KEY
    if preset_key not in CHECKPOINT_PRESETS:
        raise ValueError(f'알 수 없는 체크포인트 프리셋: {preset_key}')
    if preset_key == ACTIVE_PRESET_KEY:
        return
    ACTIVE_PRESET_KEY = preset_key
    CHECKPOINT_DIR = CHECKPOINT_PRESETS[preset_key]


# ══════════════════════════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════════════════════════
def plot_result(result: Dict, model_name: str) -> plt.Figure:
    errors   = result['errors']
    y_pred   = result['y_pred']
    thr      = result['threshold']
    segments = result['segments']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(f'Anomaly Detection — {model_name}', fontsize=14, fontweight='bold', color='#0f172a')

    # ── 위: 재구성 오차 ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#f8fafc')
    ax1.tick_params(colors='#64748b')
    ax1.xaxis.label.set_color('#64748b')
    ax1.yaxis.label.set_color('#64748b')
    ax1.title.set_color('#0f172a')
    idx = np.arange(len(errors))
    ax1.plot(idx, errors, color='#2563eb', linewidth=0.9, label='재구성 오차')
    ax1.axhline(thr, color='#f43f5e', linestyle='--', linewidth=1.2, label=f'임계값 ({thr:.5f})')
    for s in segments:
        color = '#EF4444' if s['severity'] == 'HIGH' else '#F59E0B' if s['severity'] == 'MEDIUM' else '#10B981'
        ax1.axvspan(s['start'], s['end'], alpha=0.25, color=color)
    ax1.set_ylabel('재구성 오차 (MSE)')
    ax1.set_title(f'재구성 오차 | 이상 구간: {len(segments)}개 | 이상 비율: {result["anomaly_ratio"]}%')
    ax1.grid(True, alpha=0.35, color='#cbd5e1')
    ax1.legend(fontsize=9, facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#334155')

    # ── 아래: 이상 탐지 결과 ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#f8fafc')
    ax2.tick_params(colors='#64748b')
    ax2.xaxis.label.set_color('#64748b')
    ax2.yaxis.label.set_color('#64748b')
    ax2.title.set_color('#0f172a')
    colors_pred = ['#f87171' if p == 1 else '#34d399' for p in y_pred]
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
    ax2.legend(handles=patches, fontsize=9, facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#334155')
    ax2.grid(True, alpha=0.35, axis='y', color='#cbd5e1')
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')

    plt.tight_layout()
    return fig


def plot_result_plotly(result: Dict, model_name: str) -> go.Figure:
    """
    Plotly 차트 — 브라우저에서 확대/축소/팬(도구 모음) 가능. matplotlib 정적 플롯보다 대시보드에 적합.
    """
    errors = result['errors']
    y_pred = result['y_pred']
    thr = result['threshold']
    segments = result['segments']
    idx = np.arange(len(errors), dtype=float)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.07,
        subplot_titles=(
            f'재구성 오차 | 이상 구간 {len(segments)}개 | 이상 비율 {result["anomaly_ratio"]}%',
            '윈도우별 이상 여부',
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=errors,
            mode='lines',
            name='재구성 오차',
            line=dict(color='#2563eb', width=1.2),
            hovertemplate='윈도우 %{x:.0f}<br>MSE %{y:.5f}<extra></extra>',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[idx[0], idx[-1]],
            y=[thr, thr],
            mode='lines',
            name=f'임계값 ({thr:.5f})',
            line=dict(color='#f43f5e', width=2, dash='dash'),
            hovertemplate=f'임계 {thr:.5f}<extra></extra>',
        ),
        row=1,
        col=1,
    )
    for s in segments:
        fc = (
            'rgba(239,68,68,0.28)' if s['severity'] == 'HIGH'
            else 'rgba(245,158,11,0.28)' if s['severity'] == 'MEDIUM'
            else 'rgba(16,185,129,0.28)'
        )
        fig.add_vrect(
            x0=float(s['start']) - 0.5,
            x1=float(s['end']) + 0.5,
            fillcolor=fc,
            layer='below',
            line_width=0,
            row=1,
            col=1,
        )

    bar_colors = ['#f87171' if p == 1 else '#34d399' for p in y_pred]
    fig.add_trace(
        go.Bar(
            x=idx,
            y=y_pred,
            marker_color=bar_colors,
            name='이상=1',
            showlegend=False,
            hovertemplate='윈도우 %{x:.0f}<br>이상=%{y:.0f}<extra></extra>',
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        height=640,
        margin=dict(l=52, r=28, t=72, b=48),
        title=dict(text=f'Anomaly Detection — {model_name}', font=dict(color='#0f172a', size=16), x=0.5),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=11, color='#334155')),
        hovermode='x unified',
        dragmode='zoom',
        uirevision='constant',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e2e8f0', zeroline=False, color='#64748b', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#e2e8f0', zeroline=False, color='#64748b', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='#e2e8f0', title_text='윈도우 인덱스', color='#64748b', row=2, col=1)
    fig.update_yaxes(
        range=[-0.15, 1.15],
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['정상', '이상'],
        showgrid=True,
        gridcolor='#e2e8f0',
        color='#64748b',
        row=2,
        col=1,
    )
    fig.update_annotations(font=dict(color='#475569', size=12))
    return fig


def build_dashboard_kpi_html(result: Dict) -> str:
    """제미나이 예시와 유사: KPI 카드 + Top-3 막대 (실데이터 기반, 지표는 참고용)."""
    ar = float(result['anomaly_ratio'])
    nw = int(result['n_windows'])
    na = int(result['n_anomaly'])
    ns = len(result['segments'])
    thr = float(result['threshold'])

    if ar < 2.0 and ns == 0:
        st_text, st_cls = '정상', 'dash-st-ok'
    elif ar < 12.0:
        st_text, st_cls = '주의', 'dash-st-warn'
    else:
        st_text, st_cls = '점검', 'dash-st-bad'

    health = max(0, min(100, int(100 - ar * 1.1 - ns * 2.5)))

    top3 = result.get('top3_sensors') or []
    bars_html = ''
    if top3:
        mx = max(v for _, v in top3) or 1e-12
        colors = ['#f87171', '#fb923c', '#fbbf24']
        for i, (name, v) in enumerate(top3):
            pct = min(100.0, 100.0 * float(v) / mx)
            c = colors[i % len(colors)]
            nm = html.escape(str(name))
            bars_html += (
                f'<div class="dash-bar"><div class="dash-bar-label"><span>{nm}</span>'
                f'<span class="mono">{v:.6f}</span></div>'
                f'<div class="dash-bar-track"><div class="dash-bar-fill" style="width:{pct:.1f}%;background:{c}"></div></div></div>'
            )
    else:
        bars_html = '<p class="dash-muted">이상 윈도우 없음 — Top-3 미산출</p>'

    return f"""<div class="dash-wrap">
<div class="dash-kpi-grid">
  <div class="dash-card">
    <div class="dash-card-h">가동 건강도 (참고)</div>
    <div class="dash-card-v">{health}<span class="sub">/100</span></div>
    <div class="dash-card-note">이상 비율·구간 수로 단순 환산</div>
  </div>
  <div class="dash-card">
    <div class="dash-card-h">탐지 상태</div>
    <div class="dash-card-v {st_cls}">{html.escape(st_text)}</div>
    <div class="dash-card-note">이상 윈도우 {ar:.1f}% · 구간 {ns}개</div>
  </div>
  <div class="dash-card">
    <div class="dash-card-h">윈도우</div>
    <div class="dash-card-v">{nw}<span class="sub">개</span></div>
    <div class="dash-card-note">이상 {na}개</div>
  </div>
  <div class="dash-card">
    <div class="dash-card-h">재구성 임계 (MSE)</div>
    <div class="dash-card-v dash-mono">{thr:.5f}</div>
    <div class="dash-card-note">percentile 초과 시 이상</div>
  </div>
</div>
<div class="dash-top3">
  <div class="dash-top3-title">기여도 Top-3 · 재구성 MSE (스케일 공간)</div>
  {bars_html}
  <p class="dash-muted" style="margin-top:0.75rem;font-size:0.72rem;">물리적 고장 단정 아님 · 센서 후보만 표시</p>
</div>
</div>"""


def _dashboard_error_html(message: str) -> str:
    return f'<div class="dash-wrap"><div class="dash-err">{html.escape(message)}</div></div>'


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


def build_top3_table(top3: List[Tuple[str, float]]) -> pd.DataFrame:
    """이상 윈도우에서 평균낸 피처별 재구성 오차 기준 상위 3개 (물리적 Root Cause 단정 아님)."""
    if not top3:
        return pd.DataFrame({'안내': ['이상 윈도우 없음 — Top-3 미산출']})
    return pd.DataFrame([
        {'순위': i + 1, '센서': name, '평균 MSE (스케일 공간)': round(v, 6)}
        for i, (name, v) in enumerate(top3)
    ])


# ══════════════════════════════════════════════════════════════════
#  Gradio 이벤트 핸들러
# ══════════════════════════════════════════════════════════════════
def run_analysis(
    file,
    checkpoint_preset: str,
    model_choice: str,
    model_type: str,
    threshold_pct: int,
    generate_llm: bool,
):
    set_checkpoint_preset(checkpoint_preset)

    # ── 데이터 로드 ──────────────────────────────────────────────
    try:
        if file is None:
            sample_path = os.path.join(DATA_DIR, 'sensor.csv')
            if os.path.exists(SAMPLE_UPLOAD):
                df = pd.read_csv(SAMPLE_UPLOAD, index_col=0)
            elif os.path.exists(sample_path):
                df = pd.read_csv(sample_path, nrows=5000, index_col=0)
            else:
                empty_top3 = pd.DataFrame({'안내': ['—']})
                return (
                    None,
                    _dashboard_error_html('CSV 경로 없음. 업로드하거나 sample 배치.'),
                    pd.DataFrame({'안내': ['CSV 업로드 또는 data/raw/sensor.csv / data/sample_upload/ 배치 필요.']}),
                    empty_top3,
                    '',
                )
        else:
            df = pd.read_csv(file.name, index_col=0)
    except Exception as e:
        empty_top3 = pd.DataFrame({'안내': ['—']})
        return (
            None,
            _dashboard_error_html(f'데이터 로드 오류: {e}'),
            pd.DataFrame({'에러': [str(e)]}),
            empty_top3,
            f'데이터 로드 오류: {e}',
        )

    # ── 추론 ─────────────────────────────────────────────────────
    try:
        result = run_inference(
            df,
            model_name=model_choice,
            model_type=model_type,
            checkpoint_dir=CHECKPOINT_DIR,
            threshold_pct=threshold_pct,
        )
    except FileNotFoundError as e:
        empty_top3 = pd.DataFrame({'안내': ['—']})
        return (
            None,
            _dashboard_error_html(str(e)),
            pd.DataFrame({'에러': [str(e)]}),
            empty_top3,
            str(e),
        )
    except Exception as e:
        empty_top3 = pd.DataFrame({'안내': ['—']})
        return (
            None,
            _dashboard_error_html(f'추론 오류: {e}'),
            pd.DataFrame({'에러': [str(e)]}),
            empty_top3,
            f'추론 오류: {e}',
        )

    # ── 시각화 (Plotly → 브라우저에서 확대/축소 가능) ────────────
    fig = plot_result_plotly(result, model_choice)

    # ── 세그먼트 테이블 ──────────────────────────────────────────
    seg_table = build_segment_table(result['segments'])
    top3_table = build_top3_table(result.get('top3_sensors') or [])

    # ── LLM 리포트 ───────────────────────────────────────────────
    report_text = ''
    if generate_llm:
        try:
            from rag_pipeline import generate_report
            metrics = result.get('metrics', {'f1': 'N/A', 'roc_auc': 'N/A', 'pr_auc': 'N/A'})
            report_text = generate_report(
                result['segments'],
                metrics,
                f'{model_choice} ({_model_type_label(model_type)})',
                top3_sensors=result.get('top3_sensors'),
            )
            # 리포트 저장
            report_path = os.path.join(REPORTS_DIR, 'latest_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        except Exception as e:
            report_text = f'리포트 생성 오류: {e}'

    summary = (
        f'**분석 완료** | 프리셋: `{checkpoint_preset}` | 모델: {model_choice} ({_model_type_label(model_type)}) | '
        f'총 윈도우: {result["n_windows"]} | '
        f'이상 윈도우: {result["n_anomaly"]} ({result["anomaly_ratio"]}%) | '
        f'이상 구간: {len(result["segments"])}개'
    )
    t3 = result.get('top3_sensors') or []
    if t3:
        names = ', '.join(n for n, _ in t3)
        summary += f'\n\n**Top-3 의심 센서** (이상 윈도우에서 평균 재구성 오차 기준): `{names}` — 물리적 고장 원인 단정은 아님.'
    else:
        summary += '\n\n**Top-3 의심 센서:** 이상 윈도우 없음.'

    met = result.get('metrics')
    if met:
        summary += (
            f'\n\n**라벨 기준 지표 (윈도우 단위)** — Precision={met.get("precision", "N/A")}, '
            f'Recall={met.get("recall", "N/A")}, F1={met.get("f1", "N/A")}, '
            f'ROC-AUC={met.get("roc_auc", "N/A")}, PR-AUC={met.get("pr_auc", "N/A")}'
        )
        summary += (
            f'\n**혼동** — TN={met["tn"]}, FP={met["fp"]}, FN={met["fn"]}, TP={met["tp"]} '
            f'(이상 라벨 {met["n_pos_windows"]}윈도우 / 정상 라벨 {met["n_neg_windows"]}윈도우)'
        )
        summary += f'\n*{met.get("metrics_note", "")}*'

    sn = result.get('schema_note') or ''
    if sn:
        summary = summary + f'\n\n{sn}'

    return (
        fig,
        build_dashboard_kpi_html(result),
        seg_table,
        top3_table,
        f'{summary}\n\n{report_text}',
    )


# ══════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════
def _preset_dropdown_choices():
    order = ('all', 'vif', 'legacy')
    return [(PRESET_LABELS[k], k) for k in order if k in CHECKPOINT_PRESETS]


def on_checkpoint_preset_change(preset_key: str):
    if preset_key == 'all':
        return gr.update(value='LSTM-AE')
    if preset_key == 'vif':
        return gr.update(value='CNN1D-AE')
    return gr.update()


def _sync_thr_slider_to_num(slider_val: float) -> int:
    return int(round(slider_val))


def _sync_num_to_slider(num_val):
    if num_val is None:
        return gr.update()
    v = int(round(float(num_val)))
    v = max(80, min(99, v))
    return gr.update(value=v)


def _load_sample_csv_path():
    """샘플 파일 경로를 File 컴포넌트에 주입."""
    if os.path.isfile(SAMPLE_UPLOAD):
        return gr.update(value=SAMPLE_UPLOAD)
    return gr.update()


def create_ui():
    default_preset = _default_preset_key()
    default_model = _default_model_for_preset(default_preset)

    # 기본 라이트 테마 (일반 사용자 가독성)
    _theme = gr.themes.Default(primary_hue='teal', neutral_hue='slate')

    with gr.Blocks(
        title='Pump Sensor Anomaly Detection',
        css=APP_UI_CSS,
        theme=_theme,
    ) as demo:

        gr.Markdown(APP_INTRO_MARKDOWN)

        # ── 블록 1: 사이드바 | 센서 CSV · 용어 (한 행, 좌우 분할) ──
        with gr.Row(equal_height=False, elem_classes=['app-workspace-row']):
            # ── 좌측: 분석 설정 (고정 폭 느낌의 사이드바) ──
            with gr.Column(scale=1, min_width=300, elem_classes=['app-sidebar']):
                gr.Markdown(
                    '<div style="margin-bottom:0.75rem;">'
                    '<span style="font-weight:800;letter-spacing:0.06em;font-size:1.05rem;color:#0f172a;">PUMP AI SENSE</span><br/>'
                    '<span style="font-size:0.72rem;color:#64748b;">비지도 AE · 재구성 오차</span></div>'
                )
                gr.Markdown('### ⚙️ 분석 설정')
                checkpoint_preset = gr.Dropdown(
                    label='체크포인트 프리셋',
                    info='학습된 가중치(.pt)·스케일러·피처 목록이 들어 있는 **폴더 세트**입니다. CSV 컬럼 구성과 맞는 항목을 고르세요.',
                    choices=_preset_dropdown_choices(),
                    value=default_preset,
                )
                model_choice = gr.Dropdown(
                    label='모델',
                    choices=['CNN1D-AE', 'LSTM-AE'],
                    value=default_model,
                )
                model_type = gr.Dropdown(
                    label='모델 타입',
                    info='불러올 `.pt` 파일 접미사와 같습니다. (예: `CNN1D-AE_tuned.pt`)',
                    choices=MODEL_TYPE_CHOICES,
                    value='tuned',
                )
                threshold_pct = gr.Slider(
                    label='임계값 (percentile)',
                    minimum=80,
                    maximum=99,
                    value=93,
                    step=1,
                )
                threshold_num = gr.Number(
                    value=93,
                    minimum=80,
                    maximum=99,
                    step=1,
                    precision=0,
                    label='또는 직접 입력 (80–99)',
                )
                generate_llm = gr.Checkbox(
                    label='LLM 진단 리포트 (Gemini / Vertex)',
                    value=True,
                )
                run_btn = gr.Button(
                    '🚀 분석 실행',
                    variant='primary',
                    size='lg',
                    elem_id='app-run-analysis',
                )

            # ── 우측: 업로드 · 용어만 (분석 결과는 아래 별도 풀폭 블록) ──
            with gr.Column(scale=4, min_width=280, elem_classes=['app-input-panel']):
                gr.Markdown('### 📁 센서 데이터')
                file_input = gr.File(
                    label='CSV (미선택 시 자동 샘플)',
                    file_types=['.csv'],
                    file_count='single',
                )
                sample_btn = gr.Button('샘플 CSV 불러오기', variant='secondary')

                with gr.Accordion('용어 설명 — 재구성 오차 · 임계값 · 이상 구간 · Top-3 · RAG', open=False):
                    gr.Markdown(APP_GLOSSARY_MARKDOWN)

        # ── 블록 2: 분석 결과만 — 상단 행과 분리, 가로 100% (차트·KPI·표·리포트 각각 구역) ──
        with gr.Column(elem_classes=['app-analysis-results-root']):
            gr.Markdown('### 분석 결과')
            gr.Markdown(
                '*차트는 Plotly입니다. 상단 도구로 **박스 드래그 확대**, 홈 아이콘으로 되돌리기가 가능합니다.*'
            )

            with gr.Column(elem_classes=['app-result-section', 'app-analysis-stack']):
                gr.Markdown('#### 차트 · 재구성 오차')
                plot_output = gr.Plot(label='재구성 오차 & 윈도우 이상 여부')

            with gr.Column(elem_classes=['app-result-section']):
                gr.Markdown('#### 요약 대시보드 · Top-3')
                kpi_panel = gr.HTML(value=DASHBOARD_EMPTY_HTML)

            with gr.Column(elem_classes=['app-result-section']):
                gr.Markdown('##### 이상 구간 표')
                table_output = gr.Dataframe(label='이상 구간 목록')

            with gr.Column(elem_classes=['app-result-section']):
                gr.Markdown('##### Top-3 표')
                top3_output = gr.Dataframe(
                    label='Top-3 (이상 윈도우 · 피처별 평균 MSE)',
                )

            with gr.Column(elem_classes=['app-result-section']):
                gr.Markdown('#### RAG 진단 리포트')
                report_output = gr.Markdown(show_label=False)

        checkpoint_preset.change(
            fn=on_checkpoint_preset_change,
            inputs=[checkpoint_preset],
            outputs=[model_choice],
        )

        threshold_pct.change(
            fn=_sync_thr_slider_to_num,
            inputs=[threshold_pct],
            outputs=[threshold_num],
        )
        threshold_num.change(
            fn=_sync_num_to_slider,
            inputs=[threshold_num],
            outputs=[threshold_pct],
        )

        sample_btn.click(fn=_load_sample_csv_path, inputs=[], outputs=[file_input])

        run_btn.click(
            fn=run_analysis,
            inputs=[file_input, checkpoint_preset, model_choice, model_type, threshold_pct, generate_llm],
            outputs=[plot_output, kpi_panel, table_output, top3_output, report_output],
            show_progress='full',
        )

        with gr.Accordion('사용 방법 · 체크포인트 · 환경변수', open=False):
            gr.Markdown("""
**프리셋 폴더:** `models/checkpoints_all/`(51센서·LSTM 권장) 또는 `models/checkpoints_vif/`(21센서·CNN 권장). 각 폴더에 `model_serving_config.json`, `scaler.pkl`, `final_features.pkl`, `*_tuned.pt` 등.

**컬럼:** 해당 프리셋의 `final_features.pkl`과 이름이 일치해야 합니다. `machine_status` / `label` / `timestamp`는 선택.

**샘플 CSV:** `data/sample_upload/pump_sensor_sample.csv` — 전체 스키마 기준. VIF 프리셋이면 21센서에 맞는 CSV가 필요합니다.

**LLM:** `.env`에 `GEMINI_API_KEY` 등. **`CHECKPOINT_PRESET`** = `vif` / `all` / `legacy`.

**Top-3:** 이상 윈도우에서 피처별 평균 재구성 오차 순 — 부품 고장 단정 아님.
            """)

    return demo


# ══════════════════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'Checkpoint presets: {CHECKPOINT_PRESETS}')
    print(f'Active preset: {ACTIVE_PRESET_KEY} -> {CHECKPOINT_DIR}')

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
    )
