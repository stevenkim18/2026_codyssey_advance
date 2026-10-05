# 미션: AI가 어떻게 학습하는지 수학으로 직접 풀어보기

> 개인 필수 미션 · AI/SW 심화 · AI 수학 · 60시간

## 1. 미션 소개

선형대수, 미적분, 확률통계를 NumPy로 구현하고 Matplotlib/Seaborn으로 시각화한다. 행렬 변환, 고유값 분해, 연쇄 법칙 기반 역전파, 경사하강법, 손실 함수와 확률 분포의 관계를 직접 탐구한다.

## 2. 최종 결과물

- `src/linear_algebra.py`: 회전·스케일링·전단 시각화, Power Iteration, SVD 이미지 압축/복원
- `src/calculus.py`: 중심차분 수치 미분, Gradient 시각화
- `notebooks/backprop_derivation.ipynb`: 2층 신경망 순전파/역전파 유도와 NumPy 검증
- `src/optimizer.py`: VanillaGD, Momentum 및 수렴 경로 시각화
- `notebooks/probability_loss.ipynb`: 확률분포와 MSE/CE-MLE 연결
- `README.md`: 프로젝트 개요, 실행 방법, 주요 결과

## 3. 과제 목표

- 행렬 곱셈과 행렬식의 기하학적 의미 설명
- 고유값·고유벡터와 SVD의 압축 원리 이해
- 연쇄 법칙으로 2층 신경망 역전파 손계산
- MSE와 Cross-Entropy의 확률적 의미 및 MLE 연결 설명
- Learning Rate와 Momentum의 수렴 영향 비교

## 4. 기능 요구 사항

### 선형대수

- 단위 원에 회전 `R(θ)`, 스케일링 `S(2, 0.5)`, 전단 `Sh(k)` 적용 및 전후 비교
- `det(A)`와 변환 전후 면적비 오차 1% 이내 확인
- Power Iteration 최대 고유값과 `np.linalg.eig` 결과 오차 5% 이내 검증
- SVD 이미지 복원을 `k=10, 50, 100`으로 비교

### 미적분과 역전파

- `f(x)=x²`, `x=3` 수치 미분값과 정답 6의 오차를 `1e-4` 이내로 검증
- `f(x,y)=x²+y²` 등고선과 수직인 Gradient 화살표 표시
- 구조: 입력 2 → 은닉 2 → 출력 1, Sigmoid, BCE
- 고정값: `x=[1,0]`, `W1=[[0.1,0.2],[0.3,0.4]]`, `b1=[0,0]`, `W2=[0.5,0.6]`, `b2=0`
- `z1`, `a1`, `z2`, `y_pred` 및 모든 필수 기울기와 shape 기록
- 손계산과 NumPy 결과를 소수점 4자리까지 일치시킨다.

### 최적화와 확률통계

- Vanilla GD로 `f(x,y)=x²+y²` 최적화: `lr=0.1`, 100회 후 반경 0.1 이내
- `lr>=0.5` 발산과 Momentum(`beta=0.9`) 비교
- `f(x,y)=x²+10y²`에서 Momentum 이점 확인
- 정규분포 2종 PDF, 베르누이 2종 PMF 비교
- Softmax 출력 합의 1.0 오차 `1e-6` 이내
- MSE=정규분포 MLE, Cross-Entropy=베르누이/카테고리 MLE 수식 유도

## 5. 보너스 과제

- Adam 옵티마이저와 3종 수렴 속도 비교
- Hessian 기반 Newton Method
- 엔트로피, KL-Divergence, Cross-Entropy 직접 구현

## 6. 개발 환경

- Python 3.8 이상
- NumPy 1.21+, Matplotlib 3.4+ 권장
- `np.random.seed(42)` 고정, 결과 PNG는 `outputs/`에 저장

## 7. 제약 사항

- 구현 계산은 NumPy, 시각화는 Matplotlib/Seaborn 사용
- `linalg.eig`는 Power Iteration 검증에만 사용
- PyTorch, TensorFlow, JAX 등 자동 미분 라이브러리 금지
- scikit-learn PCA·최적화 함수 금지
- 모든 함수/클래스 Docstring, 수식 주석/Markdown, `requirements.txt` 작성

