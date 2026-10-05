# 미션: AI가 어디서 자꾸 틀리는지 찾아내서 더 똑똑하게 만들기

> 개인 필수 미션 · AI/SW 심화 · 딥러닝 · 80시간

## 1. 미션 소개

Few-shot 이미지 전이학습과 RNN 시계열 예측을 수행하면서 베이스라인, 편향/분산, 오차 분석, 데이터 누수를 체계적으로 진단한다. 두 트랙 모두 베이스라인→딥러닝→진단→개선→성과 정량화 순서로 진행한다.

## 2. 최종 결과물

- Transfer Learning 및 RNN 모델 소스
- 편향/분산 진단과 해결 전략 리포트
- 오분류 최소 30건의 실패 원인 태깅 표
- 딥러닝과 베이스라인 비교표
- 필수 시각화 4종: Train/Val Loss, 예측/실제, 베이스라인 막대, 오분류 갤러리
- 시계열 분할·전처리 누수 방지 근거

## 3. 기능 요구 사항

### Few-shot 이미지 전이학습

- 클래스당 50장 미만 환경에서 ResNet/EfficientNet 등 사전학습 CNN 사용
- Linear Probing과 전체 Fine-tuning 비교
- 오분류를 사람이 직접 확인하고 최소 30건 태깅
- 권장 태그: 데이터 품질, 배경/잡음, 클래스 유사성, 라벨 오류, 모델 한계

### 시계열 예측

- LSTM/GRU로 주가·경제·전력·날씨 등 예측
- Naive, SMA, EMA, 선형회귀 중 최소 2개 베이스라인 선행 구현
- MAE, RMSE, MAPE와 딥러닝의 개선율 비교
- 동일 Test 구간에서 모든 모델 공정 비교
- 최소 1개 티커, 3년 일봉 데이터
- Train/Validation/Test=7/1/2 시간순 분할

### 편향/분산 진단

- Train/Validation Loss 곡선과 Gap 분석
- 둘 다 높으면 High Bias→모델 복잡도 증가
- Train 낮고 Validation 높으면 High Variance→정규화·데이터 증가

### 누수 방지

- shuffle 없이 시간순 분할
- `Scaler.fit()`은 Train에서만, 나머지는 `transform()`만
- 입력 윈도우에 미래 값 금지
- 지연값·이동평균 피처 생성 시점과 근거 문서화

## 4. 데이터 가이드

- 이미지: Dogs vs Cats, Food-101 일부, Oxford Flowers 일부, CIFAR-10 일부
- 시계열: yfinance, FRED, UCI/Kaggle 전력, Open-Meteo
- 금융 시계열 Jittering/Time-warping은 특성 훼손 위험으로 금지/주의

## 5. 보너스 과제

- Random Search 또는 Coarse-to-Fine 하이퍼파라미터 탐색
- LSTM/GRU Attention 또는 Transformer 비교

## 6. 개발 환경과 제약

- Python 3.10+, PyTorch 또는 TensorFlow
- GPU 권장, yfinance는 인터넷 필요
- 오차 분석은 Human-in-the-loop로 실제 실패 데이터를 확인
- 베이스라인은 딥러닝보다 먼저 측정
- 문서 용어는 Train/Validation/Test로 통일

