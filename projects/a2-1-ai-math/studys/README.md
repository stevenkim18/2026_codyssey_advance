# A2-1 AI 수학: STEP 1~6 전체 Markdown 학습자료

> 코디세이 A2-1 · Python 기초 문법을 아는 학습자를 위한 자체 학습 노트

## 빠른 시작

```bash
python -m pip install numpy matplotlib
```

각 단원은 개념 → 수식 → NumPy 실습 → 그래프 → 연습문제 → 정답/힌트 → 미션 평가 체크 순서로 구성했습니다.
실행 예제는 코드 블록을 Python 파일 또는 Jupyter Notebook에 복사해 실행하세요. 그래프는 코드 실행 후 나타나며 Markdown만 열었을 때 자동 생성되지는 않습니다.

## 전체 목차 (총 32개 소단원)

### STEP 1. 선형대수

- [1-1 벡터와 행렬](STEP_1_선형대수/STEP_1-1_벡터와_행렬.md)
- [1-2 선형 변환](STEP_1_선형대수/STEP_1-2_선형_변환.md)
- [1-3 행렬식](STEP_1_선형대수/STEP_1-3_행렬식.md)
- [1-4 고유값과 고유벡터](STEP_1_선형대수/STEP_1-4_고유값과_고유벡터.md)
- [1-5 특이값 분해(SVD)](STEP_1_선형대수/STEP_1-5_특이값_분해_SVD.md)

### STEP 2. 미적분

- [2-1 함수와 기울기](STEP_2_미적분/STEP_2-1_함수와_기울기.md)
- [2-2 극한과 미분](STEP_2_미적분/STEP_2-2_극한과_미분.md)
- [2-3 편미분](STEP_2_미적분/STEP_2-3_편미분.md)
- [2-4 Gradient(기울기 벡터)](STEP_2_미적분/STEP_2-4_Gradient기울기_벡터.md)
- [2-5 NumPy 수치 미분과 Gradient 검증하기](STEP_2_미적분/STEP_2-5_NumPy_수치_미분과_Gradient_검증하기.md)

### STEP 3. 신경망과 역전파

- [3-1 퍼셉트론과 인공신경망](STEP_3_신경망_역전파/STEP_3-1_퍼셉트론과_인공신경망.md)
- [3-2 가중치와 편향](STEP_3_신경망_역전파/STEP_3-2_가중치와_편향.md)
- [3-3 활성화 함수](STEP_3_신경망_역전파/STEP_3-3_활성화_함수.md)
- [3-4 순전파(Forward Propagation)](STEP_3_신경망_역전파/STEP_3-4_순전파Forward_Propagation.md)
- [3-5 연쇄 법칙(Chain Rule)](STEP_3_신경망_역전파/STEP_3-5_연쇄_법칙Chain_Rule.md)
- [3-6 역전파(Backpropagation) 직접 구현](STEP_3_신경망_역전파/STEP_3-6_역전파Backpropagation_직접_구현.md)

### STEP 4. 최적화

- [4-1 손실 함수(Loss Function)](STEP_4_최적화/STEP_4-1_손실_함수Loss_Function.md)
- [4-2 경사하강법(Gradient Descent)](STEP_4_최적화/STEP_4-2_경사하강법Gradient_Descent.md)
- [4-3 학습률(Learning Rate)](STEP_4_최적화/STEP_4-3_학습률Learning_Rate.md)
- [4-4 Momentum(모멘텀)](STEP_4_최적화/STEP_4-4_Momentum모멘텀.md)
- [4-5 학습 과정 시각화 및 비교](STEP_4_최적화/STEP_4-5_학습_과정_시각화_및_비교.md)

### STEP 5. 확률과 통계

- [5-1 확률 기본](STEP_5_확률_통계/STEP_5-1_확률_기본.md)
- [5-2 확률분포·평균·분산](STEP_5_확률_통계/STEP_5-2_확률분포·평균·분산.md)
- [5-3 가능도(Likelihood)](STEP_5_확률_통계/STEP_5-3_가능도Likelihood.md)
- [5-4 최대우도추정(MLE)](STEP_5_확률_통계/STEP_5-4_최대우도추정MLE.md)
- [5-5 MSE와 Cross-Entropy](STEP_5_확률_통계/STEP_5-5_MSE와_Cross-Entropy.md)
- [5-6 확률적 손실함수 이해(NLL 연결)](STEP_5_확률_통계/STEP_5-6_확률적_손실함수_이해NLL_연결.md)

### STEP 6. 통합 실습과 제출

- [6-1 선형대수 실습 정리](STEP_6_통합_제출/STEP_6-1_선형대수_실습_정리.md)
- [6-2 수치 미분·역전파 검증](STEP_6_통합_제출/STEP_6-2_수치_미분·역전파_검증.md)
- [6-3 경사하강법·Momentum 실험](STEP_6_통합_제출/STEP_6-3_경사하강법·Momentum_실험.md)
- [6-4 결과 그래프·해석 정리](STEP_6_통합_제출/STEP_6-4_결과_그래프·해석_정리.md)
- [6-5 Notebook·README 작성 및 제출 점검](STEP_6_통합_제출/STEP_6-5_Notebook·README_작성_및_제출_점검.md)

## A2-1 평가 핵심 조건

- **선형대수**: 회전/스케일링/전단 3종 그래프, 행렬식 면적비 오차 1% 이내, Power Iteration 오차 5% 이내, SVD k=10·50·100 한 Figure
- **미적분**: 수치미분과 해석해의 절대오차 1e-4 이내, Gradient와 등고선의 수직 관계를 시각화
- **신경망/역전파**: 2→2→1 Sigmoid+BCE 모델의 순전파·손실·역전파·Gradient Checking
- **최적화**: 학습률별 수렴/발산, GD vs Momentum 비교
- **확률**: Probability/Likelihood/MLE, MSE·BCE·CE와 음의 로그 가능도 연결

## STEP 6 추천 실행 순서

1. [STEP 6-1 선형대수 실습 정리](STEP_6_통합_제출/STEP_6-1_선형대수_실습_정리.md)에서 그래프 3종, 면적비, 고유값, SVD를 확인합니다.
2. [STEP 6-2 수치 미분·역전파 검증](STEP_6_통합_제출/STEP_6-2_수치_미분·역전파_검증.md)에서 미분과 2층 신경망을 점검합니다.
3. [STEP 6-3 경사하강법·Momentum 실험](STEP_6_통합_제출/STEP_6-3_경사하강법·Momentum_실험.md)에서 그래프 PNG를 저장합니다.
4. [STEP 6-4 결과 그래프·해석 정리](STEP_6_통합_제출/STEP_6-4_결과_그래프·해석_정리.md)로 실험 요약을 만듭니다.
5. [STEP 6-5 Notebook·README 작성 및 제출 점검](STEP_6_통합_제출/STEP_6-5_Notebook·README_작성_및_제출_점검.md)에서 제출 체크리스트를 마무리합니다.

## 주의사항

- 이 자료는 학습 및 평가 준비용입니다. 공식 미션 지침에 새 변경 사항이 있다면 공식 공지를 우선합니다.
- `np.random.default_rng(42)` 등의 고정 시드를 사용한 결과는 사용 환경의 수치오차에 따라 마지막 자리 수가 다를 수 있습니다.
- 압축파일을 해제한 뒤 상대 경로를 유지해야 문서 간 링크가 연결됩니다.
- 정답/힌트는 GitHub에서 `<details>`를 클릭하면 펼쳐집니다.
