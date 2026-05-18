# 산업용 펌프 센서 이상 탐지 시스템

> LSTM/CNN Autoencoder + Optuna 자동 튜닝 + RAG 기반 LLM 진단 리포트를 결합한 비지도 이상 탐지 파이프라인

---

## 개요

52개 센서 시계열 데이터에서 **정상 패턴만 학습**한 Autoencoder가 재구성 오차를 기반으로 이상 구간을 탐지하고, Gemini LLM이 RAG를 통해 자연어 진단 리포트를 생성합니다.

```
sensor CSV 업로드
      ↓
schema_validation  →  컬럼 정합 검사
      ↓
inference.py       →  이상 탐지 + Top-3 의심 센서
      ↓
rag_pipeline.py    →  ChromaDB 검색 + Gemini 리포트
      ↓
Gradio UI          →  Plotly 차트 + KPI + 리포트 출력
```

---

## 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | Kaggle — pump_sensor_data |
| 크기 | 220,320행 / 52개 센서 / 2018-04~08 |
| 레이블 | NORMAL 93.4% / RECOVERING 6.6% / BROKEN 0.003% (7건) |
| 특성 | 극단적 불균형 → 지도 학습 불가 → 비지도 설계 |

원본 `sensor.csv`는 용량 문제로 Git 미포함. 로컬에 직접 배치 필요.

---

## 설계 결정

| 결정 | 내용 | 이유 |
|---|---|---|
| 평가 지표 | **F2 Score (β=2)** | 미탐(FN) 비용 ≫ 오탐(FP) — Recall 최우선 |
| 전처리 | bfill 완전 제거 → ffill → fillna(0) | 실시간 환경에서 미래 참조(Leakage) 차단 |
| 분할 | 시간순 55 / 15 / 30% | stratify 없음 — 시계열 순서 보존 |
| Optuna | Val F2 최대화 / 500 trials | 설계 목표와 일관성 유지 |
| A/B 실험 | VIF 21피처 vs 전체 51피처 | 피처 축소가 항상 유리하지 않음을 실증 |

---

## 아키텍처

### 모델 비교 (5종)

| 모델 | 종류 |
|---|---|
| LSTM-AE | 시계열 Autoencoder |
| CNN1D-AE | 1D Conv Autoencoder |
| Transformer-AE | Self-Attention Autoencoder |
| Isolation Forest | 트리 기반 비지도 |
| One-Class SVM | 경계 기반 비지도 |

### 최종 선정 결과

| 버전 | 선정 모델 | Val F2 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| All-sensors (51피처) | **LSTM-AE** | 0.4441 | **0.9091** | 0.9294 |
| VIF (21피처) | CNN1D-AE | 0.4112 | 0.7222 | 0.8209 |

챔피언: All-sensors LSTM-AE (미탐 방지 설계 목표와 일치)

### Optuna 튜닝 결과

| 버전 | 튜닝 전 Val F2 | 튜닝 후 Val F2 | 향상 |
|---|---|---|---|
| VIF CNN1D-AE | 0.1104 | 0.4112 | +0.30 |
| All LSTM-AE | 0.2292 | 0.4441 | +0.21 |

---

## RAG 파이프라인

```
피처별 MSE 역산
      ↓
Top-3 의심 센서 추출
      ↓
ChromaDB 검색 (pump_regulations.txt)
      ↓
Gemini 2.5 Flash 프롬프트 주입
      ↓
자연어 진단 리포트 생성
      ↓
light_grounding_check (환각 방지 가드레일, PoC)
```

---

## 프로젝트 구조

```
sensor_anomaly/
├── app.py                    # Gradio UI (포트 7860)
├── inference.py              # AE 추론, Top-3 MSE, 윈도우 지표
├── rag_pipeline.py           # Chroma RAG + Gemini 리포트
├── serving_config.py         # model_serving_config.json 로드
├── schema_validation.py      # CSV 컬럼 검증
├── requirements.txt
│
├── data/
│   ├── raw/sensor.csv        # 원본 (Git 제외)
│   └── sample_upload/        # 시연용 5,000행 샘플
│
├── models/
│   ├── checkpoints_all/      # 프리셋 `all` — 51센서 LSTM-AE 권장
│   └── checkpoints_vif/      # 프리셋 `vif` — 21센서 CNN1D-AE 권장
│
├── notebooks/
│   ├── clean/                # 메인 파이프라인 (EDA + 학습 + 시각화)
│   ├── kaggle/               # Kaggle 제출용 (시각화 축소)
│   └── val_compare/          # default vs tuned Val 비교
│
├── rag/
│   ├── pump_regulations.txt  # 규정 원문 (RAG 소스)
│   └── chroma_db/            # 벡터 DB (실행 시 자동 생성, Git 제외)
│
└── scripts/                  # 노트북 자동 빌더
```

---

## 실행 방법

### 환경 설정

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### .env 설정

```
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=asia-northeast3
CHECKPOINT_PRESET=all        # all / vif / legacy
```

### 앱 실행

```bash
python app.py
# → http://127.0.0.1:7860
```

### 체크포인트 프리셋 매칭

| 프리셋 | 폴더 | CSV 컬럼 수 | 권장 모델 |
|---|---|---|---|
| `all` | checkpoints_all/ | 51개 | LSTM-AE |
| `vif` | checkpoints_vif/ | 21개 | CNN1D-AE |

> `data/sample_upload/pump_sensor_sample.csv`는 51센서용입니다. VIF 프리셋 사용 시 컬럼 불일치 에러가 발생할 수 있습니다.

---

## GPU 최적화

| 기법 | 효과 |
|---|---|
| AMP (Mixed Precision) | 속도 1.5~2배, Tensor Core 활용 |
| Window Cache 사전 생성 | 500 trials 윈도우 재생성 병목 제거 |
| pin_memory=False, workers=0 | CUDA DataLoader 충돌 방지 |

Optuna 500 trials 기준: CPU 약 58시간 → GPU + AMP + Cache 약 3시간대 (약 95% 단축)

---

## 주요 트러블슈팅

| 번호 | 문제 | 원인 | 해결 |
|---|---|---|---|
| TS-01 | VIF_THRESH=10 → 피처 7개, F1=0 | 산업 센서 공선성 과소평가 | 임계값 10→50 완화 |
| TS-02 | y_test anomaly 0% | 이상이 앞 80%에 집중 | 시간순 55/15/30 분할 전환 |
| TS-04 | CPU Optuna 500 trials = 58시간 | trial당 420초 × 500 | GPU + AMP + Cache 적용 |
| TS-10 | Optuna trial inf → 전체 중단 | 기울기 폭주 | nan_to_num + catch=(ValueError,) |
| TS-17 | 튜닝 후 Test F2 < default Val | Val 최적화 ≠ Test 일반화 | 설계 특성으로 해석·보고서 명시 |

---

## 기술 스택

`Python` `PyTorch` `Optuna` `Gradio` `Plotly` `ChromaDB` `Vertex AI Gemini` `Scikit-learn` `Pandas` `NumPy`

---

## 한계 및 향후 계획

- Val 최적화와 Test 일반화 괴리 — 임계값 F2 기반 동적 최적화 예정
- Top-3 센서는 재구성 오차 기여도 후보일 뿐, 물리적 고장 원인과 다를 수 있음
- RAG `light_grounding_check`는 PoC 수준 — Hybrid Search (BM25 + Vector) 교체 예정
- Vertex AI 임베딩 교체로 RAG 검색 품질 향상 예정
- Kafka 실시간 스트리밍 파이프라인 연동 예정
