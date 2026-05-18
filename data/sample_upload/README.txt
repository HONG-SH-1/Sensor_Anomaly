pump_sensor_sample.csv
======================

- 내용: data/raw/sensor.csv 의 앞쪽 5,000행(원본과 동일한 컬럼 스키마).
- 용도: Gradio에서 파일 없이 실행할 때, 또는 업로드 테스트용.

중요
----

1) 추론 시 반드시 맞아야 하는 것
   - 체크포인트 폴더의 final_features.pkl 에 들어 있는 센서 컬럼 이름·개수와
     업로드 CSV가 일치해야 합니다.
   - 노트북에서 sensor_15 등을 제거했다면, 학습 후 저장된 final_features.pkl 기준으로
     샘플을 맞추거나, 동일 전처리를 거친 CSV를 준비해야 합니다.

2) train/val/test 로 나눈 중간 산출물(npy 등)을 넣을 필요는 없습니다.
   - 앱/ inference.py 는 원시에 가까운 시계열 CSV를 받아
     학습 때와 같이 ffill만(실시간 가정으로 bfill 없음) → fillna(0) → 저장된 scaler.transform → 윈도우 로 처리합니다.

3) 비지도 학습
   - AE는 정상 패턴 재학습이 핵심입니다. machine_status / label 은 있으면 지표용이며
     없어도 재구성 오차 기반 탐지는 동작합니다.

4) 재현성
   - 학습(Kaggle)과 동일한 SEED, 패키지 버전, 전처리 순서를 맞추는 것이
     “완벽에 가까운” 실험에 해당합니다. 추론 단계에서는 저장된 scaler.pkl 을
     재학습하지 않고 transform 만 합니다.
