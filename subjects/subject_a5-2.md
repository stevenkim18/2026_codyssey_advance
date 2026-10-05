# 미션: AI가 왜 그런 결정을 했는지 이유를 사람에게 설명해주게 하기

> 개인 필수 미션 · AI/SW 심화 · 머신러닝 · 80시간

## 1. 미션 소개

K-Means로 고객 페르소나를 만들고, 분류 모델의 승인·거절 판단을 SHAP으로 설명한다. 군집화→분류→Global/Local XAI를 하나의 분석 흐름으로 연결해 기술 결과를 비즈니스 언어로 전달한다.

## 2. 최종 결과물

- PCA 2차원 군집 산점도와 군집별 통계·페르소나
- SHAP Summary Plot PNG
- 승인/거절 각 1건의 Waterfall PNG 또는 Force Plot HTML 캡처
- 상위 변수 1개 이상의 Dependence Plot PNG와 인사이트 2문장
- `README.md`: 군집 정의, 해석, 군집별 전략과 리스크 관리

## 3. 케이스 정의

- 거절: `is_overdue=1` 예측 확률이 0.5 이상인 고위험 고객
- 승인: `is_overdue=0` 예측 확률이 높은 저위험 고객
- 군집별 대표 고객 1~2명을 선정해 Local SHAP과 페르소나를 연결하는 것을 권장

## 4. 기능 요구 사항

- A5-1에서 생성한 `finance_data.csv`를 그대로 사용
- 군집 입력에서 `credit_score`, `is_overdue` 등 타겟 제외
- 스케일링, 결측치, 이상치 처리는 허용하되 임의 컬럼·외부 데이터·증강 금지
- K-Means 전 스케일링 수행
- Elbow와 Silhouette Score로 K를 선정하고 해석 가능성도 고려
- PCA 산점도에 Explained Variance Ratio 표시
- 군집별 통계적 특징과 페르소나 정의
- Random Forest 등 Tree 모델에 SHAP 적용
- `TreeExplainer` 우선, `KernelExplainer` 지양
- Summary Plot, Local Waterfall/Force Plot, Dependence Plot 생성
- Summary 상위 1~2개 및 비즈니스 중요 변수 선택
- 그래프별 비즈니스 해석과 군집별 실행 전략 작성

## 5. README 권장 구조

1. 문제와 목표
2. 군집 요약표
3. 군집별 페르소나
4. Global SHAP 상위 3개 변수의 방향과 의미
5. 승인/거절 Local 사례
6. 대상·메시지·액션 중심 실행 제언
7. 한계와 추가 실험

## 6. 보너스 과제

- Streamlit 입력→확률·SHAP 대시보드
- DBSCAN 또는 Hierarchical Clustering과 K-Means 비교

