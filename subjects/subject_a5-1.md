# 미션: 대출을 해줘도 될지 AI가 대신 판단해주는 시스템 만들기

> 개인 필수 미션 · AI/SW 심화 · 머신러닝 · 80시간

## 1. 미션 소개

가상 금융 데이터로 규칙 기반 베이스라인과 머신러닝을 비교하고, 신용 점수 회귀와 연체 위험 분류를 하나의 파이프라인으로 구현한다. 데이터 누수, 불균형, 규제, 앙상블과 임계값이 금융 의사결정에 미치는 영향을 분석한다.

## 2. 최종 결과물

- `finance_data.csv` 10,000건 기반 규칙/ML 성능 비교표: Accuracy, F1, AUC
- 6개 입력 변수로 0~1000 신용 점수를 예측하는 회귀 모델
- Alpha별 계수 변화 그래프와 RMSE·MAE·R² 중 2개 이상
- 연체 확률, Confusion Matrix, ROC 곡선과 AUC
- `README.md`, `requirements.txt`, GitHub 저장소

## 3. 기능 요구 사항

- 제공 코드를 `data_gen.py`로 실행하되 생성 데이터와 불균형을 임의 수정하지 않는다.
- `.gitignore`로 `*.csv`를 저장소에서 제외한다.
- 최소 5개 if-else 규칙으로 베이스라인을 구현한다.
- 베이스라인 Accuracy, Precision, Recall, F1 측정
- `Pipeline` 또는 `ColumnTransformer`로 수치형 대치·스케일링과 범주형 인코딩 모듈화
- 전처리·스케일러·인코더·SMOTE는 Train에서만 fit
- Train/Test 8:2, `random_state=42`, 5-fold CV 권장
- Ridge와 Lasso, Alpha `0.01, 0.1, 1, 10, 100` 비교
- `class_weight='balanced'` 또는 SMOTE로 불균형 처리
- Random Forest 또는 XGBoost와 GridSearchCV 적용
- 탐색 조합은 100개 이하 권장
- 특징 중요도 시각화와 3줄 해석 권장

## 4. 보너스 과제

- `BaseEstimator` 기반 Custom Transformer
- 추가 파생변수 생성과 성능 비교
- 비용 민감 임계값 최적화 등 확장 실험

## 5. 핵심 데이터 컬럼

- `age`, `annual_income`, `spending_score`, `debt_ratio`
- `credit_card_count`, `overdue_count_6m`
- 회귀 타겟 `credit_score`, 분류 타겟 `is_overdue`
- 데이터는 `N_SAMPLES=10000`, `RANDOM_STATE=42`로 생성

## 6. 검토 체크리스트

- 규칙 기반과 ML 모델의 동일 지표 비교
- 회귀/분류 데이터 분할 및 평가 구분
- Train 외 구간 정보로 전처리를 fit하지 않았는지 확인
- 불균형 대응 전후 결과 기록
- 재현 가능한 실행 명령과 라이브러리 버전 명시

