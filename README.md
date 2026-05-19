# 산업용 펌프 센서 이상 탐지 시스템 (수정중)

**Python** · **PyTorch** · **Optuna** · **Gradio** · **ChromaDB** · **Vertex AI Gemini**

극단적 클래스 불균형(고장 7건 / 22만 행) 환경에서 **미탐(FN) 비용이 오탐(FP)보다 크다**는 전제 아래, F2 Score(β=2)를 설계 목표로 삼은 비지도 이상 탐지 파이프라인이다. LSTM/CNN Autoencoder가 정상 패턴만 학습해 재구성 오차로 이상을 탐지하고, Optuna로 하이퍼파라미터를 자동 튜닝한다. **All-sensors(51피처) vs VIF(21피처)** A/B 실험으로 “피처 축소 = 항상 유리”라는 가정을 검증한 뒤, 챔피언 모델 위에 ChromaDB RAG + Gemini가 Top-3 의심 센서 기반 진단 리포트를 생성한다.

---

## 아키텍처

### End-to-End 파이프라인

```
sensor CSV 업로드
      ↓
schema_validation     →  final_features.pkl 컬럼 정합 검사
      ↓
inference.py          →  AE 재구성 오차 · 이상 탐지 · Top-3 의심 센서
      ↓
rag_pipeline.py       →  ChromaDB 검색 + Gemini 진단 리포트
      ↓
Gradio UI (app.py)     →  Plotly 차트 · KPI · 리포트 출력
```

### A/B 실험 트랙 (모델 선정)

동일한 평가 프로토콜(시간순 55/15/30% 분할 · Val F2 선정 · Test는 참고)로 두 전처리 트랙을 병렬 비교했다.

```
원본 CSV (52센서)
      ↓
공통 전처리: ffill → fillna(0)  [bfill 금지 — 미래 참조 차단]
      ↓
┌─────────────────────────────┬─────────────────────────────┐
│  Track A — All-sensors      │  Track B — VIF              │
│  51피처 (sensor_15 제외)    │  VIF≤50 → 21피처            │
│  5종 벤치마크               │  5종 벤치마크               │
│  선정: LSTM-AE ✅           │  선정: CNN1D-AE ✅          │
│  Optuna 500 trials          │  Optuna 500 trials          │
└─────────────────────────────┴─────────────────────────────┘
      ↓
챔피언: All-sensors LSTM-AE (Test Recall 0.9091 — 미탐 방지 목표와 일치)
      ↓
RAG: Top-3 MSE 역산 → pump_regulations.txt 검색 → Gemini 리포트
```

**All-sensors LSTM-AE를 챔피언으로 둔 이유:** VIF 트랙은 Val F2·튜닝 효율은 양호하나, 최종 Test Recall이 0.7222로 All(0.9091) 대비 낮다. 산업 펌프 맥락에서 “의심 구간을 놓치지 않는 것”이 우선이므로 51피처·LSTM-AE를 서빙 기본값(`CHECKPOINT_PRESET=all`)으로 둔다.

### RAG 진단 블록

```
피처별 MSE 역산
      ↓
Top-3 의심 센서 추출
      ↓
ChromaDB 검색 (rag/pump_regulations.txt)
      ↓
Gemini 2.5 Flash 프롬프트 주입
      ↓
자연어 진단 리포트 생성
      ↓
light_grounding_check (환각 방지 가드레일, PoC)
```

### 핵심 설계 결정

| 결정 | 내용 | 이유 |
|---|---|---|
| 평가 지표 | **F2 Score (β=2)** | 미탐(FN) 비용 ≫ 오탐(FP) — Recall 최우선 |
| 전처리 | bfill 완전 제거 → ffill → fillna(0) | 실시간 환경에서 미래 참조(Leakage) 차단 |
| 분할 | 시간순 55 / 15 / 30% | stratify 없음 — 시계열 순서 보존 |
| Optuna | Val F2 최대화 / 500 trials | 설계 목표와 일관성 유지 |
| A/B 실험 | VIF 21피처 vs 전체 51피처 | 피처 축소가 항상 유리하지 않음을 실증 |

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| AI/ML | Python 3.10+, PyTorch, Optuna, Scikit-learn |
| 데이터 | Pandas, NumPy, SciPy, Statsmodels |
| 대시보드 | Gradio, Plotly, Matplotlib, Seaborn |
| RAG/LLM | ChromaDB, LangChain, Google GenAI, Vertex AI Gemini |
| 임베딩 | sentence-transformers |
| 개발 환경 | VS Code (Cursor), Kaggle Notebook |
| 버전 관리 | Git, GitHub |

---

## 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | [Kaggle — pump_sensor_data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) |
| 크기 | 220,320행 / 52개 센서 / 2018-04~08 |
| 레이블 | NORMAL 93.4% / RECOVERING 6.6% / BROKEN 0.003% (7건) |
| 특성 | 극단적 불균형 → 지도 학습 불가 → **비지도 Autoencoder** 설계 |

**전처리 파이프라인**

1. `sensor_15` 등 결측·상수 컬럼 제거 (트랙별 상이)
2. **bfill 금지** → `ffill` → `fillna(0)` (실시간 가정)
3. Train 구간만 `MinMaxScaler.fit` → Val/Test는 `transform`만
4. Sliding Window → 3D 텐서 (모델별 window_size는 Optuna·serving JSON에 저장)

원본 `data/raw/sensor.csv`는 용량 문제로 Git에 포함하지 않는다. Kaggle에서 받아 `data/raw/`에 배치한다. 시연용 `data/sample_upload/pump_sensor_sample.csv`(5,000행)는 저장소에 포함되어 있다.

---

## 벤치마크 결과

동일 분할·F2(β=2) 기준으로 5종 모델을 비교했다. **모델 선정은 Val F2**, 아래 Test 지표는 참고용이다.

### Track A — All-sensors (51피처)

| 모델 | Val F2 | 학습 시간 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| **LSTM-AE** ✅ | **0.2292** | 140.8s | 0.9231 | 0.9442 |
| Transformer-AE | 0.2027 | 344.6s | 0.9231 | 0.9913 |
| CNN1D-AE | 0.1887 | 86.9s | 0.9231 | 0.9412 |
| Isolation Forest | — | 0.6s | 0.8462 | 0.8953 |
| One-Class SVM | — | 7.6s | 0.8462 | 0.9013 |

**선정 근거:** Val F2 1위(LSTM-AE). Transformer는 Test ROC-AUC는 높으나 Val F2·학습 시간(344.6s) 대비 이득이 없고, CNN1D는 Val F2가 낮다. 1위 모델만 Optuna 500 trials 대상으로 삼는다.

### Track B — VIF (21피처, VIF≤50)

| 모델 | Val F2 | 학습 시간 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| **CNN1D-AE** ✅ | **0.1104** | 84.6s | 0.8462 | 0.9234 |
| LSTM-AE | 0.0839 | 134.1s | 0.7692 | 0.8732 |
| Transformer-AE | 0.0475 | 271.6s | 0.9231 | 0.9711 |
| Isolation Forest | — | 0.6s | — | — |
| One-Class SVM | — | 2.4s | — | — |

**선정 근거:** VIF 축소 구간에서는 CNN1D-AE가 Val F2·학습 속도 균형이 가장 좋다. 다만 **전체 챔피언은 Track A LSTM-AE** (Test Recall 0.9091 vs 0.7222).

---

## 최종 성능 (Optuna 튜닝 후)

### All-sensors — LSTM-AE

| 단계 | Val F2 | Test F2 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| Default (튜닝 전) | 0.2292 | 0.1567 | 0.9231 | 0.9442 |
| **Optuna Best** ✅ | **0.4441** | 0.0709 | **0.9091** | 0.9294 |

튜닝으로 Val F2는 +0.21 상승했으나, Test F2는 하락한다(Val 최적화 ≠ Test 일반화 — TS-17). **미탐 방지 목표** 관점에서는 Test Recall 0.9091 유지가 핵심이므로 튜닝 가중치를 서빙에 반영한다.

**Best Params (요약):** hidden=32, latent=8, window=30, threshold_pct=90, dropout=0.05

### VIF — CNN1D-AE

| 단계 | Val F2 | Test F2 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| Default (튜닝 전) | 0.1104 | 0.1436 | 0.8462 | 0.9234 |
| **Optuna Best** ✅ | **0.4112** | 0.0888 | 0.7222 | 0.8209 |

**Best Params (요약):** hidden=128, latent=8, window=100, threshold_pct=90, dropout=0.05

---

## Gradio 시연

| 구분 | 내용 |
|---|---|
| 실행 | `python app.py` → http://127.0.0.1:7860 |
| 샘플 데이터 | `data/sample_upload/pump_sensor_sample.csv` (5,000행, 51센서) |
| 기본 프리셋 | `CHECKPOINT_PRESET=all` → LSTM-AE_tuned 권장 |

**Kaggle Test 홀드아웃 vs UI 시연**

| 지표 | All LSTM-AE (튜닝 후) | VIF CNN1D-AE (튜닝 후) | 비고 |
|---|---|---|---|
| Test Recall | **0.9091** | 0.7222 | 챔피언 선정 근거 |
| Test F2 | 0.0709 | 0.0888 | 불균형으로 F2·F1은 낮게 나옴 |
| Test ROC-AUC | 0.9294 | 0.8209 | 점수 기반 랭킹은 양호 |

샘플 CSV는 연속 5,000행 추출본으로, **BROKEN 라벨이 거의 없어** UI에서 “고장 확정” 데모는 제한적이다. AE는 정상 패턴 재학습 + 재구성 오차 기반이므로 `machine_status` 없이도 탐지는 동작하며, 라벨은 KPI·검증용이다.

Top-3 센서는 **재구성 오차 기여도 상위 후보**일 뿐, 물리적 고장 원인과 1:1 대응하지 않을 수 있다.

---

## 준비

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

1. [Kaggle 데이터셋](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)에서 `sensor.csv`를 받아 `data/raw/sensor.csv`에 둔다.
2. 학습 산출물(`.pt`, `.pkl`)을 각 체크포인트 폴더에 배치한다. 가중치·스케일러는 `.gitignore` 대상이며, 노트북 학습 후 로컬에 생성한다.
3. RAG·LLM 사용 시 프로젝트 루트에 `.env`를 둔다.

---

## 실행

### Gradio 대시보드

```bash
python app.py
```

처음 실행 시 사용하는 프리셋 폴더에 아래 파일이 있어야 한다.

| 파일 | 프리셋 | 설명 |
|---|---|---|
| `model_serving_config.json` | all / vif | 윈도우·임계·Optuna 하이퍼 (저장소 포함) |
| `scaler.pkl` | all / vif | Train fit MinMaxScaler |
| `final_features.pkl` | all / vif | 추론용 센서 컬럼 목록 |
| `LSTM-AE_tuned.pt` | **all** (권장) | 51센서 챔피언 가중치 |
| `CNN1D-AE_tuned.pt` | **vif** | 21센서 VIF 트랙 가중치 |
| `*_default.pt`, `*_optuna_best.pt` | 선택 | UI에서 모델 종류 전환 시 |

### .env 예시

```
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=asia-northeast3
CHECKPOINT_PRESET=all        # all / vif / legacy
# GEMINI_API_KEY=...         # RAG 리포트 사용 시
```

### 체크포인트 프리셋

| 프리셋 | 폴더 | CSV 센서 수 | 권장 모델 |
|---|---|---|---|
| `all` | `models/checkpoints_all/` | 51 | LSTM-AE |
| `vif` | `models/checkpoints_vif/` | 21 | CNN1D-AE |

> `pump_sensor_sample.csv`는 51센서용이다. `CHECKPOINT_PRESET=vif`일 때 컬럼 불일치가 날 수 있다.

---

## 노트북 실행 순서

노트북은 Kaggle·로컬을 상단 환경 감지 코드로 구분한다. `data/raw/sensor.csv`가 필요하다.

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `notebooks/clean/all_clean.ipynb` | All-sensors EDA · 5종 벤치마크 · LSTM Optuna · 체크포인트 저장 |
| 2 | `notebooks/clean/vif_clean.ipynb` | VIF 피처 선택 · 5종 벤치마크 · CNN Optuna |
| 3 | `notebooks/clean/visualization_only.ipynb` | 학습 로그·체크포인트 기반 시각화 (figures) |
| 4 | `notebooks/kaggle/all_kaggle.ipynb` | Kaggle 제출용 (clean 축소본, `scripts/build_kaggle_noviz.py` 생성) |
| 5 | `notebooks/kaggle/vif_kaggle.ipynb` | VIF Kaggle 제출용 |
| 6 | `notebooks/val_compare/val_compare.ipynb` | default vs tuned Val 지표 비교 (`scripts/build_val_compare_nb.py` 생성) |

`outputs/figures/`, `outputs/reports/` 산출물은 용량상 Git 미포함.

---

## 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `app.py` | Gradio UI (포트 7860) |
| `inference.py` | AE 추론, Top-3 MSE, 윈도우 지표 |
| `rag_pipeline.py` | Chroma RAG + Gemini 리포트 |
| `serving_config.py` | `model_serving_config.json` 로드 |
| `schema_validation.py` | 업로드 CSV 컬럼 검증 |
| `data/raw/sensor.csv` | 원본 (Git 제외) |
| `data/sample_upload/pump_sensor_sample.csv` | 시연용 5,000행 |
| `models/checkpoints_all/` | 프리셋 `all` — 51센서 LSTM-AE |
| `models/checkpoints_vif/` | 프리셋 `vif` — 21센서 CNN1D-AE |
| `notebooks/clean/` | 메인 학습·EDA 파이프라인 |
| `notebooks/kaggle/` | Kaggle 제출용 노트북 |
| `notebooks/val_compare/` | Val default vs tuned 비교 |
| `rag/pump_regulations.txt` | RAG 규정 원문 |
| `rag/chroma_db/` | 벡터 DB (실행 시 생성, Git 제외) |
| `scripts/` | 노트북 자동 빌더 |

---

## GPU 최적화

| 기법 | 효과 |
|---|---|
| AMP (Mixed Precision) | 속도 1.5~2배, Tensor Core 활용 |
| Window Cache 사전 생성 | 500 trials 윈도우 재생성 병목 제거 |
| pin_memory=False, workers=0 | CUDA DataLoader 충돌 방지 |

Optuna 500 trials 기준: CPU 약 58시간 → GPU + AMP + Cache 약 3시간대 (약 95% 단축, TS-04)

---

## 주요 트러블슈팅

| # | 문제 현상 | 원인 분석 | 해결 방안 |
|---|---|---|---|
| TS-01 | VIF_THRESH=10 → 피처 7개, F1=0 | 산업 센서 공선성 과소평가 | 임계값 10→50 완화 |
| TS-02 | y_test anomaly 0% | 이상이 앞 80%에 집중 | 시간순 55/15/30 분할 전환 |
| TS-04 | CPU Optuna 500 trials ≈ 58시간 | trial당 420초 × 500 | GPU + AMP + Window Cache |
| TS-10 | Optuna trial inf → 전체 중단 | 기울기 폭주 | nan_to_num + catch=(ValueError,) |
| TS-17 | 튜닝 후 Test F2 < default | Val 최적화 ≠ Test 일반화 | 설계 특성으로 문서화·Recall 우선 보고 |

---

## 한계점 및 향후 로드맵

- **Val vs Test 괴리:** 임계값·F2를 Test 구간 피드백으로 재조정하는 동적 최적화 검토.
- **Top-3 센서:** MSE 기여도 후보이며, 물리 고장 원인과 다를 수 있음 — 도메인 전문가 검증 병행.
- **RAG 품질:** `light_grounding_check`는 PoC — Hybrid Search(BM25 + Vector), Vertex 임베딩 교체 예정.
- **실시간 연동:** CSV 데모를 넘어 Kafka 등 스트리밍 파이프라인 연동.
- **데이터 다양성:** 단일 Kaggle 펌프 데이터 의존 완화, 다른 설비·센서 스키마 검증.

---

## 라이선스·데이터 사용

- **데이터셋:** [pump-sensor-data (Kaggle)](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) — 업로더·Kaggle 이용약관 및 데이터 라이선스를 따른다. 원본 CSV는 이 저장소에 포함하지 않는다.
- **코드:** 학습·포트폴리오 참고용으로 공개한다.
