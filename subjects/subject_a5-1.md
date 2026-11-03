# A5-1. 대출을 해줘도 될지 AI가 대신 판단해주는 시스템 만들기

> 개인 필수 미션 · AI/SW 심화 · 머신러닝 · 80시간

## 문제 기술

### 1. 미션 소개

머신러닝 파이프라인은 데이터 전처리, 모델 학습, 성능 평가를 하나의 논리적 흐름으로 연결하는 자동화 시스템이다. 금융권의 대출 심사, 이상 거래 탐지, 제조업의 불량품 선별 등 신뢰성이 중요한 다양한 산업에서 활용된다.

당신은 시중은행의 신용리스크 담당자로서, 고객의 대출 승인/거절/보류 정책을 결정하는 AI 모델을 구축해야 한다. 모델의 예측 결과가 고객의 금융 생활에 직접적 영향을 미치므로, 성능 지표의 선택과 임계값 설정이 비즈니스 의사결정과 직결된다.

이 미션에서는 가상의 금융 데이터(신용 점수 및 연체 기록)를 직접 생성하고, 규칙 기반 베이스라인과 머신러닝 모델의 성능을 비교하는 이중 예측 모델(회귀/분류)을 구현하며 지도학습의 전체 파이프라인을 경험한다. 단순히 라이브러리를 사용하는 것이 아니라, "규칙 기반 시스템 대비 학습이 왜 효과적인가"를 정량적 결과로 확인한다.

특히, 같은 데이터에서 문제 유형이 달라지면(회귀 vs 분류) 지표, 모델, 해석이 어떻게 달라지는지를 체감하는 것이 핵심 학습 목표이다. 또한, 데이터 불균형 처리와 규제(Regularization) 적용을 통해 안정적인 성능을 확보하는 원리를 이해하고, 앙상블 기법으로 단일 모델의 한계를 극복하는 설계 방식을 경험한다.

이 미션에서 다루는 ML 파이프라인 역량은 금융 영역에 국한되지 않는다. 이커머스의 고객 이탈 예측, 제조업의 불량품 검출, 헬스케어의 위험도 예측, 모빌리티의 수요 예측 등 다양한 도메인으로 전이 가능하며, 이후 진행될 모델 해석(24번) 및 딥러닝 최적화 주제로 확장될 수 있다.

### 2. 최종 결과물

다음 기능이 정상 동작하는 분석 시스템을 구현하고, GitHub 레포지토리(Repository) URL을 제출한다.

1. **베이스라인 vs 머신러닝 비교 시스템**
   - 입력 : `finance_data.csv` (10,000건의 고객 데이터)
   - 출력 : 규칙 기반 모델과 ML 모델의 성능 비교표 (Accuracy, F1-Score, AUC)

2. **신용 점수 예측 모델 (회귀)**
   - 입력 : 고객의 소득, 부채 비율, 연체 이력 등 6개 변수
   - 출력 : 0~1000 범위의 신용 점수 예측값, 규제 강도(Alpha)별 계수 변화 그래프
   - 필수 평가 지표 : RMSE, MAE, R² 중 2개 이상 산출 및 비교표 포함

3. **연체 위험 예측 모델 (분류)**
   - 입력 : 동일한 고객 데이터
   - 출력 : 연체 확률 (0~1), Confusion Matrix, ROC-AUC 곡선

4. **프로젝트 문서 및 관리 파일**
   - `README.md` : 프로젝트 개요, 실행 방법, 성능 비교 리포트
   - `requirements.txt` : 프로젝트 실행에 필요한 라이브러리 및 버전 명세

### 3. 과제 목표

이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

- 규칙 기반 시스템 대비 머신러닝의 장점과 한계를 정량적 근거와 함께 설명할 수 있다.
- 지도 학습 파이프라인의 단계별 역할과 데이터 누수(Data Leakage) 방지 원리를 설명할 수 있다.
- 회귀와 분류 문제의 차이를 이해하고 적절한 평가지표를 구분할 수 있다.
- 과대적합을 해결하기 위한 규제(L1/L2)의 동작 방식을 이해한다.
- 데이터 불균형 문제 해결을 위한 리샘플링 및 가중치 조정 방식을 설명할 수 있다.

## 기능 요구 사항

다음 요구사항을 모두 만족해야 한다.

### 1. 데이터 생성 및 Git 환경 구성

- 제공된 [데이터 생성 코드]를 `data_gen.py`로 작성하고 실행하여 `finance_data.csv`가 생성되어 있어야 한다.
- 생성된 데이터에는 의도적인 노이즈와 불균형이 포함되어 있으므로, 임의로 데이터를 수정하지 않는다.
- `git init` 후 `.gitignore`를 설정하여 데이터 파일(`*.csv`)이 저장소에 업로드되지 않도록 관리해야 한다.

### 2. 규칙 기반 베이스라인 구현

- `if-else` 조건문을 활용한 규칙 기반 예측 모델이 최소 5개 이상의 규칙으로 정의되어 있어야 한다.
- 베이스라인 모델의 Accuracy, Precision, Recall, F1-Score가 측정되어 있어야 한다.
- 베이스라인 vs ML 모델의 성능 비교표가 `README.md`에 포함되어 있어야 한다.

### 3. 데이터 파이프라인 (Data Pipeline)

- Scikit-learn의 `Pipeline` 또는 `ColumnTransformer`를 사용하여 전처리 과정이 모듈화되어 있어야 한다.
- 수치형 변수는 대치(Imputation) 후 스케일링을, 범주형 변수는 인코딩이 적용되어 있어야 한다.
- 스케일러와 인코더의 fit은 반드시 Train Set에서만 수행되어 있어야 한다.
- 데이터 분할 및 재현성 체크리스트
  - Train/Test 분할 비율: 8:2 (stratified random split 권장)
  - `random_state=42` 고정 필수
  - Cross-Validation: 5-fold 권장
  - 전처리/SMOTE/스케일링은 반드시 Train Set에서만 fit

### 4. 모델링 및 성능 최적화

- 회귀 모델 : Ridge와 Lasso를 모두 적용하고, Alpha 값 변화(0.01, 0.1, 1, 10, 100)에 따른 계수 변화 그래프가 있어야 한다.
- 회귀 모델의 RMSE, MAE, R² 중 2개 이상이 측정되어 있어야 한다.
- 분류 모델 : `class_weight='balanced'` 또는 SMOTE가 적용되어 불균형이 처리되어 있어야 한다.
- 분류 모델의 Confusion Matrix 시각화가 있어야 한다.
- ROC-AUC 곡선이 시각화되어 있고 AUC 값이 명시되어 있어야 한다.
- 앙상블 : Random Forest 또는 XGBoost가 적용되고 GridSearchCV로 하이퍼파라미터가 최적화되어 있어야 한다.
  - GridSearchCV 탐색 가이드 : 파라미터당 후보 값 5개 이하 권장, 총 탐색 조합 100개 이하 권장
- (권장) 앙상블 모델의 특징 중요도(Feature Importance) 시각화와 해석 3줄 요약을 `README.md`에 포함한다.

### [참고: 데이터 생성 코드]

아래 코드를 `data_gen.py`로 저장하고 실행하여 `finance_data.csv`를 생성한다.

이 코드는 신용 점수(회귀 타겟)와 연체 여부(분류 타겟)를 동시에 생성한다.

학습에 필요한 데이터 불균형과 상관관계를 의도적으로 주입하도록 설계하였다.

```python
import pandas as pd
import numpy as np
import random

# 1. 데이터 생성 설정
N_SAMPLES = 10000  # 샘플 개수
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 2. Feature 생성 (가상의 고객 정보)
data = {
    'age': np.random.randint(20, 70, N_SAMPLES),  # 나이
    'annual_income': np.random.normal(5000, 2000, N_SAMPLES).round(0), # 연 소득 (단위: 만원)
    'spending_score': np.random.randint(1, 100, N_SAMPLES), # 소비 점수
    'debt_ratio': np.random.uniform(0, 1, N_SAMPLES).round(2), # 소득 대비 부채 비율
    'credit_card_count': np.random.randint(1, 10, N_SAMPLES), # 보유 신용카드 수
    'overdue_count_6m': np.random.poisson(0.5, N_SAMPLES) # 최근 6개월 연체 횟수 (Poisson 분포)
}

df = pd.DataFrame(data)

# 소득이 음수인 경우 0으로 처리 (노이즈 수정)
df['annual_income'] = df['annual_income'].apply(lambda x: max(x, 1500))

# 3. Target 1 생성: 신용 점수 (Credit Score) - 회귀 문제용
# 규칙: 소득이 높고 연체가 적을수록 점수가 높음 + 랜덤 노이즈
df['credit_score'] = (
    300
    + (df['annual_income'] / 100) * 3
    - (df['overdue_count_6m'] * 50)
    - (df['debt_ratio'] * 100)
    + np.random.normal(0, 30, N_SAMPLES) # 노이즈 추가
)

# 신용점수 범위 제한 (0 ~ 1000점)
df['credit_score'] = df['credit_score'].clip(0, 1000).round(0)

# 4. Target 2 생성: 연체 여부 (Overdue) - 분류 문제용
# 규칙: 신용점수가 낮고 부채 비율이 높으면 연체 확률 증가 (불균형 데이터 생성)
# 기본 연체율을 낮게 설정하여 '데이터 불균형' 상황 연출
threshold = df['credit_score'].quantile(0.15) # 하위 15% 점수 기준
df['is_overdue'] = np.where(
    (df['credit_score'] < threshold) & (np.random.rand(N_SAMPLES) > 0.2),
    1, # 연체 (Positive)
    0 # 정상 (Negative)
)

# 데이터 셔플
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 5. 파일 저장
df.to_csv('finance_data.csv', index=False)

print(f"데이터 생성 완료: finance_data.csv")
print(f"전체 샘플 수: {len(df)}")
print(f"연체(1) 비율: {df['is_overdue'].mean()*100:.2f}% (불균형 데이터 확인)")
```

이 데이터의 장점 : Regression용 `credit_score`가 소득/연체와 선형 관계를 가지며, Classification용 `is_overdue` 비율이 약 10~15%로 불균형 처리의 효과를 보여줄 수 있다.

## 보너스 과제(선택)

### 1. Custom Transformer 구현

- Scikit-learn의 `BaseEstimator`를 상속받아 나만의 전처리 클래스(예: 파생변수 생성기)를 구현한다.
- 배움 포인트 : Scikit-learn의 설계 철학과 확장 방식을 이해한다.

### 2. 모듈화 (Modularization)

- Jupyter Notebook의 코드를 `.py` 스크립트 파일로 분리하여 실행 가능하도록 구조화한다.
- 배움 포인트 : 재사용 가능한 코드 구조 설계를 경험한다.

### 3. Learning Curve 분석

- 데이터 크기에 따른 Train/Validation 성능 변화를 시각화한다.
- 배움 포인트 : 편향-분산 트레이드오프를 시각적으로 체감한다.

## 참고 사항: 데이터셋 확보 가이드

아래 코드를 `data_gen.py`로 저장하고 실행하여 `finance_data.csv`를 생성한다. 이 코드는 신용 점수(회귀 타겟)와 연체 여부(분류 타겟)를 동시에 생성하며, 학습에 필요한 데이터 불균형과 상관관계를 의도적으로 주입하도록 설계했다.

```python
import pandas as pd
import numpy as np
import random

# 1. 데이터 생성 설정
N_SAMPLES = 10000  # 샘플 개수
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 2. Feature 생성 (가상의 고객 정보)
data = {
    'age': np.random.randint(20, 70, N_SAMPLES),  # 나이
    'annual_income': np.random.normal(5000, 2000, N_SAMPLES).round(0), # 연 소득 (단위: 만원)
    'spending_score': np.random.randint(1, 100, N_SAMPLES), # 소비 점수
    'debt_ratio': np.random.uniform(0, 1, N_SAMPLES).round(2), # 소득 대비 부채 비율
    'credit_card_count': np.random.randint(1, 10, N_SAMPLES), # 보유 신용카드 수
    'overdue_count_6m': np.random.poisson(0.5, N_SAMPLES) # 최근 6개월 연체 횟수 (Poisson 분포)
}

df = pd.DataFrame(data)

# 소득이 음수인 경우 0으로 처리 (노이즈 수정)
df['annual_income'] = df['annual_income'].apply(lambda x: max(x, 1500))

# 3. Target 1 생성: 신용 점수 (Credit Score) - 회귀 문제용
# 규칙: 소득이 높고 연체가 적을수록 점수가 높음 + 랜덤 노이즈
df['credit_score'] = (
    300
    + (df['annual_income'] / 100) * 3
    - (df['overdue_count_6m'] * 50)
    - (df['debt_ratio'] * 100)
    + np.random.normal(0, 30, N_SAMPLES) # 노이즈 추가
)

# 신용점수 범위 제한 (0 ~ 1000점)
df['credit_score'] = df['credit_score'].clip(0, 1000).round(0)

# 4. Target 2 생성: 연체 여부 (Overdue) - 분류 문제용
# 규칙: 신용점수가 낮고 부채 비율이 높으면 연체 확률 증가 (불균형 데이터 생성)
# 기본 연체율을 낮게 설정하여 '데이터 불균형' 상황 연출
threshold = df['credit_score'].quantile(0.15) # 하위 15% 점수 기준
df['is_overdue'] = np.where(
    (df['credit_score'] < threshold) & (np.random.rand(N_SAMPLES) > 0.2),
    1, # 연체 (Positive)
    0 # 정상 (Negative)
)

# 데이터 셔플
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 5. 파일 저장
df.to_csv('finance_data.csv', index=False)

print(f"데이터 생성 완료: finance_data.csv")
print(f"전체 샘플 수: {len(df)}")
print(f"연체(1) 비율: {df['is_overdue'].mean()*100:.2f}% (불균형 데이터 확인)")
```

**이 데이터의 장점:**

- **Regression용**: `credit_score`가 소득, 연체 횟수와 선형 관계를 가지므로 Ridge/Lasso 실습에 최적화되어 있습니다.
- **Classification용**: `is_overdue` 비율이 약 10~15% 내외로 생성되어, 불균형 처리(SMOTE 등)의 효과를 극적으로 보여줄 수 있습니다.

## 6. 개발 환경

- Python 3.10 이상

## 7. 제약 사항

- 권장 환경 : Python 3.10+ 권장
- 필수 라이브러리 : Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn만 사용한다.
- 선택 허용 라이브러리 : imbalanced-learn(SMOTE), XGBoost 사용 시 `requirements.txt`에 버전을 명시한다.
- 데이터 : 반드시 제공된 코드로 생성한 데이터를 사용하며, 외부 데이터는 사용하지 않는다.
- 형상 관리 : 기능 단위(전처리, 모델링, 문서화)로 커밋(Commit) 로그를 남겨야 한다.
- 이 과제에서는 딥러닝 프레임워크(PyTorch, TensorFlow)는 사용하지 않는다.

## 8. 결과 예시

아래는 정답이 아니라 참고 예시다. 레포지토리 구조는 달라도 된다.

### 규칙기반 베이스라인 예시

```python
def rule_based_predict(row):
    if row['overdue_count_6m'] >= 3: return 1  # 연체 3회 이상
    if row['debt_ratio'] > 0.7: return 1       # 부채비율 70% 초과
    if row['annual_income'] < 3000: return 1   # 저소득
    if row['credit_card_count'] > 7: return 1  # 카드 과다 보유
    return 0  # 정상
```

### 성능 비교표 예시

| 모델 | Accuracy | F1-Score | AUC | Recall |
| --- | ---: | ---: | ---: | ---: |
| 규칙 기반 베이스라인 | 0.78 | 0.45 | 0.65 | 0.52 |
| Logistic Regression | 0.82 | 0.58 | 0.79 | 0.63 |
| Random Forest (튜닝) | 0.87 | 0.72 | 0.89 | 0.75 |

### 회귀 성능 비교표 예시

| 모델 | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Ridge (Alpha=1) | 35.2 | 27.1 | 0.91 |
| Lasso (Alpha=1) | 36.8 | 28.5 | 0.90 |

### README.md 예시

ML 모델이 베이스라인 대비 F1-Score 약 60% 향상, AUC 약 37% 향상 Lasso 규제 강도(Alpha)를 10으로 설정했을 때, 카드 개수 변수의 가중치가 0이 되어 제거됨
