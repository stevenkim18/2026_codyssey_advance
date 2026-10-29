# STEP 3-3. 활성화 함수

> 코디세이 A2-1 AI 수학 · STEP 3 신경망과 역전파 · Python/NumPy 입문자용

## 학습 목표

- Sigmoid, ReLU, Softmax를 구분한다.
- Sigmoid 도함수를 이해한다.
- 출력층 선택이 손실 함수에 영향을 미침을 이해한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 왜 행렬 곱만 반복하면 안 될까?

선형 변환 두 번을 하면 결국 다시 하나의 선형 변환으로 합칠 수 있다. 층을 여러 개 쌓아도 중간에 비선형 처리가 없다면 표현력이 제한된다. **활성화 함수**가 이 문제를 해결한다.

### Sigmoid

\[\sigma(z)=\frac{1}{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)(1-\sigma(z))\]

숫자를 0과 1 사이로 압축한다. 이진 분류에서 출력값을 확률처럼 해석하는 데 자주 쓰인다.

### ReLU

\[\operatorname{ReLU}(z)=\max(0,z)\]

양수는 그대로, 음수는 0으로 만든다. `z=0`에서는 미분이 정의되지 않지만, 구현에서는 특정 서브그래디언트를 선택한다(여기서는 0).

### Softmax

\[\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}\]

여러 클래스 점수를 합이 1인 확률 분포로 변환한다. **입력 전체에서 최대값을 빼는 방식**이 지수 함수 오버플로를 줄인다.

## 2. NumPy 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1/(1+np.exp(-z))
def relu(z): return np.maximum(0,z)
def softmax(z):
    z = np.asarray(z, dtype=float)
    ez = np.exp(z-np.max(z))
    return ez/ez.sum()

z = np.linspace(-6, 6, 300)
plt.plot(z, sigmoid(z), label='Sigmoid')
plt.plot(z, relu(z), label='ReLU')
plt.title('Activation functions'); plt.ylim(-.2, 6.5)
plt.xlabel('z'); plt.legend(); plt.grid(alpha=.2); plt.show()

p = softmax(np.array([2.,1.,.1]))
print('Softmax:', p, 'sum=',p.sum())
assert np.isclose(p.sum(), 1.)
print('Sigmoid derivative at 0:', sigmoid(0)*(1-sigmoid(0))) # 0.25
```

## 3. 손실 함수와 짝짓기

- **이진 분류**: Sigmoid 출력 + Binary Cross-Entropy(BCE)
- **여러 클래스 중 하나 선택**: Softmax 출력 + Categorical Cross-Entropy
- **연속값 회귀**: 반드시 Sigmoid를 쓰지 않으며 선형 출력 + MSE가 흔하다.

> 주의: Sigmoid가 0 또는 1에 가까운 포화 구간에서는 도함수가 작아져 기울기 전달이 약해질 수 있다.

---

## 직접 풀어보기

① `sigmoid(0)`과 미분값을 구하라. ② `relu(-3)`, `relu(2)`는? ③ Softmax 확률의 합을 검증하라.

<details>
<summary>정답·힌트 확인</summary>

① `0.5`, `0.25`. ② `0`, `2`. ③ `np.isclose(probs.sum(), 1)`로 검증한다.

</details>

## 미션 평가 체크

- [ ] 평가용 Sigmoid + BCE 모델과 ReLU 예제를 혼동하지 않는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 3-2 가중치와 편향](STEP_3-2_가중치와_편향.md) · [전체 목차](../README.md) · [다음: 3-4 순전파(Forward Propagation) →](STEP_3-4_순전파Forward_Propagation.md)
