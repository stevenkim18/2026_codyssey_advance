# A5-2. AI가 왜 그런 결정을 했는지 이유를 사람에게 설명해주게 하기

분야: AI/SW 심화  
구분: 머신러닝  
학습시간: 80시간  
필수/선택: 필수

## 문제기술

### 기술적 설명

### 1. 미션 소개

비지도 학습과 XAI(설명 가능한 AI)는 정답이 없는 데이터에서 가치를 발견하고, 블랙박스 모델의 투명성을 확보하는 기술이다. 금융 규제 당국은 대출 거절 사유를 설명하도록 요구하고, 의료 현장에서는 진단 근거를 제시해야 한다.

이 미션의 핵심 스토리라인은 다음과 같다

- 군집화로 고객 페르소나(누구인가)를 도출
- 분류 모델로 거절/승인을 예측
- SHAP으로 '왜 그런 판정인가'를 설명하는 전 과정을 수행

이 세 단계는 하나의 연결된 분석 파이프라인으로, 군집별 대표 고객을 선정하고 Local SHAP으로 해당 페르소나의 예측 근거를 설명하는 것까지 연결된다.

이 미션에서는 군집화(Clustering)를 통해 고객의 숨겨진 페르소나를 정의하고, SHAP 기술을 활용해 모델의 예측 근거를 시각적으로 설명하는 과정을 수행한다. 단순히 알고리즘을 적용하는 것을 넘어, "왜 이런 결과가 나왔는가?"를 논리적으로 설명하는 분석 능력을 결과물로 확인한다.

또한, 분석한 내용을 바탕으로 비즈니스 인사이트를 도출하여 설득력 있는 리포트를 작성하는 역량을 기른다. 이 경험을 통해 Global Interpretation(전체 모델 해석)과 Local Interpretation(개별 예측 해석)의 차이를 이해하고, 기술적 결과를 비기술자에게 전달하는 역량을 기른다.

트랙별 적용 시나리오 안내

본 미션의 핵심 질문 "왜 이런 판정을 했는가?"는 다양한 도메인에 동일하게 적용된다:

- Mobility: "왜 이 경로/판단을 했는가?"
- Energy: "왜 이상 소비/설비 이상으로 판단했는가?"
- Healthcare: "왜 고위험군/재방문 위험으로 분류했는가?"
- E-Commerce: "왜 이탈 위험/혜택 제한으로 판단했는가?"

### 2. 최종 결과물

다음 내용이 모두 포함된 GitHub 레포지토리(Repository) URL을 제출한다.

1. 고객 군집화 분석 시스템
   - 입력 : finance_data.csv의 입력 변수 (타겟 변수 제외)
   - 출력 : PCA 2차원 산점도 (군집별 색상 구분), 각 군집의 통계적 특징 및 페르소나 정의

2. 모델 해석 - Global Interpretation (전체 해석)
   - 입력 : Mission 23에서 학습한 분류 모델 (Random Forest 등)
   - 출력 : SHAP Summary Plot (전체 변수 중요도와 방향성) - PNG 저장 필수

3. 모델 해석 - Local Interpretation (개별 해석)
   - 입력 : 특정 고객 1명의 데이터 (승인/거절 케이스 각 1건)
   - 출력 : Force Plot 또는 Waterfall Plot (해당 고객의 예측 근거) - Waterfall Plot PNG(권장) 또는 Force Plot HTML 캡처/저장(필수 중 택1)

4. SHAP Dependence Plot
   - 특정 변수 1개 이상에 대한 Dependence Plot 및 인사이트 2문장 이상 - PNG 저장 필수

5. 인사이트 리포트 (README.md)
   - 군집 정의 : 도출된 각 군집의 통계적 특징 및 페르소나(Persona) 정의
   - 해석 결과 : 모델의 주요 변수(Feature Importance) 및 특정 고객의 예측 사유 분석
   - 비즈니스 제언 : 군집별 추천 전략 및 리스크 관리 방안

- 승인/거절 케이스 선정 기준

  본 미션에서 "승인"과 "거절"은 다음과 같이 정의한다:

  - 거절 케이스: 연체(is_overdue=1) 예측 확률이5 이상인 고객 (고위험 판정)
  - 승인 케이스: 연체(is_overdue=0) 예측 확률이 높은 고객 (저위험 판정)

  군집별 대표 고객 1명(또는 2명)을 선정하여 Local SHAP 해석을 수행하고, 해당 페르소나와 연결하여 설명하는 것을 권장한다.

### 3. 과제 목표

이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

- 정답이 없는 데이터에서 거리(Distance) 기반으로 군집을 형성하는 원리를 설명할 수 있다.
- Elbow Method와 Silhouette Score를 활용하여 최적의 K를 결정하는 방법을 이해한다.
- 차원 축소(PCA)의 의미와 Explained Variance(설명된 분산)를 해석할 수 있다.
- SHAP 값을 활용해 복잡한 모델의 예측 결과를 변수별 기여도로 설명할 수 있다.
- Global Interpretation(Summary Plot: 전체 변수 영향)과 Local Interpretation(Force Plot: 개별 예측 이유)의 차이를 구분할 수 있다.
- 기술적 분석 결과를 비기술자(예: 영업팀)에게 비즈니스 언어로 전달할 수 있다.
- PCA의 주성분(PC)이 원본 특성(Feature)과 다르다는 점을 이해하고, 해석 시 혼동을 피할 수 있다.

### 4. 기능 요구 사항

다음 요구사항을 모두 만족해야 한다.

1. 데이터 준비
   - Mission 23 에서 생성한 finance_data.csv 파일을 그대로 사용해야 한다.
   - 군집화 분석 시에는 타겟 변수(신용 점수, 연체 여부)를 제외하고 입력 변수(Feature)만 사용해야 한다.
   - 데이터 사용 정책
     - 허용: 원본 CSV 기준 스케일링, 결측치 처리, 이상치 제거 등 전처리
     - 금지: 임의 컬럼 추가, 외부 데이터 병합, 데이터 증강(Augmentation)

2. 고객 군집화 (Clustering)
   - K-Means 적용 전 데이터 스케일링이 수행되어 있어야 한다.
   - Elbow Method와 Silhouette Score를 통해 최적의 군집 수(K)가 선정되어 있어야 한다.
   - K 결정 가이드: Silhouette Score를 우선 기준으로 하고, Elbow를 보조 기준으로 활용하며, 최종적으로 페르소나의 해석 가능성(명확성)을 고려하여 선정한다.
   - PCA를 적용하여 군집 결과가 2차원 산점도로 시각화되어 있어야 한다.
   - PCA 시각화에 Explained Variance Ratio(설명된 분산 비율)가 표시되어 있어야 한다.
   - 각 군집의 통계적 특징과 페르소나가 정의되어 있어야 한다.

3. 모델 해석 (XAI)
   - Tree 기반 모델(Random Forest 등)에 shap 라이브러리가 적용되어 있어야 한다.
   - TreeExplainer를 우선으로 사용하고, KernelExplainer는 지양한다.
   - Summary Plot으로 전체 변수 중요도와 방향성이 시각화되어 있어야 한다. (Global Interpretation)
   - Force Plot 또는 Waterfall Plot으로 개별 데이터(승인/거절 케이스)의 예측 원인이 시각화되어 있어야 한다. (Local Interpretation)
   - Dependence Plot으로 특정 변수 변화에 따른 예측값 추이가 시각화되어 있어야 한다.
   - 변수 선택 기준 : Summary Plot 상위 변수 1~2개 + 비즈니스 의미가 큰 변수 1개

4. 인사이트 리포트
   - 그래프의 비즈니스적 의미가 텍스트로 기술되어 있어야 한다. (그래프와 함께 비즈니스적 해석을 반드시 포함)
   - 군집별 마케팅/리스크 관리 전략 제언이 포함되어 있어야 한다.

- README.md 권장 템플릿 구조

  아래 섹션 구조를 참고하여 리포트를 작성한다 (구조는 참고이며 내용이 핵심)

  - 문제/목표 (한 문단)
  - 군집 요약 (표 1개)
  - 페르소나 정의 (군집별 3줄)
  - Global SHAP 해석 (상위 3 변수: 방향/해석)
  - Local 사례 2건 (승인/거절)
  - 실행 제언 (대상/메시지/액션)
  - 한계/추가 실험

### 5. 보너스 과제

1. 인터랙티브 대시보드 (Streamlit)
   - app.py 를 생성하여 웹 브라우저에서 고객 정보를 입력하면 이탈 확률과 SHAP 설명이 나오는 대시보드를 구현하고 레포지토리에 포함한다.
   - 배움 포인트 : 분석 결과를 실제 서비스로 배포하는 경험을 한다.

2. 군집화 알고리즘 비교
   - DBSCAN 또는 Hierarchical Clustering을 적용하고 K-Means 결과와 비교 분석한다.
   - 배움 포인트 : 알고리즘별 특성과 적합한 상황을 이해한다.

#### [참고 사항] 데이터셋 확보 가이드

아래 코드를 data_gen.py로 저장하고 실행하여 finance_data.csv를 생성한다. 이 코드는 신용 점수(회귀 타겟)와 연체 여부(분류 타겟)를 동시에 생성하며, 학습에 필요한 데이터 불균형과 상관관계를 의도적으로 주입하도록 설계했다

```css
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
    0  # 정상 (Negative)
)

# 데이터 셔플
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 5. 파일 저장
df.to_csv('finance_data.csv', index=False)

print(f"데이터 생성 완료: finance_data.csv")
print(f"전체 샘플 수: {len(df)}")
print(f"연체(1) 비율: {df['is_overdue'].mean()*100:.2f}% (불균형 데이터 확인)")
```

- **이 데이터의 장점:**
  - **Regression용:** `credit_score`가 소득, 연체 횟수와 선형 관계를 가지므로 **Ridge/Lasso** 실습에 최적화되어 있습니다.
  - **Classification용:** `is_overdue` 비율이 약 10~15% 내외로 생성되어, 불균형 처리(SMOTE 등)의 효과를 극적으로 보여줄 수 있습니다.

### 개발환경

### 6. 개발 환경

- Python 3.10 이상

### 제약조건

### 7. 제약 사항

- 라이브러리: Scikit-learn, SHAP 필수 사용. Scikit-learn 내장 모델만 허용한다.
- 허용/필수 라이브러리 요약

  | **구분** | **라이브러리** | **비고** |
  | --- | --- | --- |
  | **필수** | Scikit-learn, SHAP | 핵심 분석 라이브러리 |
  | **허용** | NumPy, Pandas, Matplotlib, Seaborn | 데이터 처리 및 시각화 |
  | **금지** | XGBoost, LightGBM 등 외부 모델링 라이브러리 | Scikit-learn 내장 모델만 사용 |

- 보고서 작성: 그래프와 함께 비즈니스적 해석을 텍스트로 반드시 포함해야 한다.
- 데이터: Mission 23에서 생성한 동일한 데이터를 사용해야 한다. (데이터 사용 정책 참조)
- 실행 환경: Python 3.10+ 권장. shap/scikit-learn 버전을txt에 명시한다.
- 실행 순서: analysis_clustering → analysis_shap → README 정리 순으로 구성한다.
- SHAP 트러블슈팅 가이드
  - initjs()는 Jupyter 환경에서 Force Plot 렌더링 시에만 필요하다.
  - Force Plot이 렌더링되지 않을 경우 Waterfall Plot을 대안으로 사용할 수 있다.
  - 데이터 1만 건 이상 시 SHAP 연산 시간이 길어질 수 있으며, 샘플링(1,000~2,000건)을 권장한다.
  - TreeExplainer를 우선으로 사용하고, KernelExplainer는 연산 비용이 높으므로 지양한다.

### Test Case

### 8. 결과 예시

아래는 정답이 아니라 참고 예시다. 레포지토리 구조는 달라도 된다.

- 군집 수 결정 예시

  ```css
  Elbow Method: K=4에서 기울기 변화 (Inertia 급감 후 완만)
  Silhouette Score: K=4일 때 0.42로 최대값 → 최적 K=4 선정
  PCA Explained Variance: PC1(45%) + PC2(28%) = 73% 설명력
  ```

- README.md 예시

  ```css
  ## 페르소나 정의
  - Cluster A (VIP): 소득 상위 10%, 부채비율 0.15 이하, 연체 이력 0회. 신용카드 활용도가 높으며 장기 거래 가능성 높음. → 프리미엄 카드 마케팅 + 대출 한도 상향 제안
  - Cluster B (주의군): 부채 비율 70% 이상, 연체 횟수 2회 이상, 소득 대비 소비 점수 높음. 과소비 패턴으로 연체 위험 높음. → 신용 관리 프로그램 안내 + 자동 결제 한도 설정 권고
  
  ## SHAP 해석 결과
  - Global: "연체 횟수"가 예측에 가장 큰 영향을 미침
  - Local (고객 #1234): 연체 횟수(+0.35), 부채 비율(+0.22)이 높아 거절 확률 80%
  ```
