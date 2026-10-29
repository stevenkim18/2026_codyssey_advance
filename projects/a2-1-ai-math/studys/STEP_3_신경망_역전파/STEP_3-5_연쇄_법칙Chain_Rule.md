# STEP 3-5. 연쇄 법칙(Chain Rule)

> 코디세이 A2-1 AI 수학 · STEP 3 신경망과 역전파 · Python/NumPy 입문자용

## 학습 목표

- 복합함수의 미분을 이해한다.
- Sigmoid+BCE에서 출력 오차 항이 어떻게 단순화되는지 유도한다.
- 기울기가 뒤에서 앞으로 전달되는 원리를 말한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 러시아 인형처럼 겹친 함수

`x`를 먼저 제곱하고 `sin`을 적용하면 `f(x)=sin(x²)`다. 바깥 함수와 안쪽 함수를 차례로 미분해 곱한다.

\[\frac{d}{dx}\sin(x^2)=\cos(x^2)\cdot 2x\]

이것이 **연쇄 법칙**이다. 신경망은 `행렬 곱 → 활성화 → 행렬 곱 → 활성화 → 손실`의 합성함수이므로 똑같이 적용된다.

## 2. 계산 그래프에서 거꾸로 추적

`z2→ŷ→L` 순서로 순전파했다면, 역전파는 `L→ŷ→z2` 순서로 미분한다.

\[\frac{\partial L}{\partial z_2}=\frac{\partial L}{\partial \hat y}\frac{\partial\hat y}{\partial z_2}\]

Sigmoid와 BCE를 사용하면

\[\frac{\partial L}{\partial\hat y}=-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}\]

이고 `∂ŷ/∂z2=ŷ(1-ŷ)`이므로 곱해서 정리하면

\[\boxed{\frac{\partial L}{\partial z_2}=\hat y-y}\]

이다. **이 간결한 식이 3-6 역전파의 출발점**이다.

## 3. 코드로 국소 미분 확인

```python
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))
def bce_from_z(z, y):
    p = np.clip(sigmoid(z), 1e-12, 1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

z, y = .3, 1.
h = 1e-5
numeric = (bce_from_z(z+h,y)-bce_from_z(z-h,y))/(2*h)
analytic = sigmoid(z)-y
print('수치미분:', numeric, '연쇄법칙:', analytic)
assert abs(numeric-analytic)<1e-5
```

## 4. 다음 층으로 기울기 보내기

출력층 `z2=a1@W2+b2`이므로

\[\frac{\partial L}{\partial W_2}=a_1^T\delta_2,\quad
\frac{\partial L}{\partial b_2}=\delta_2\]

(단일 샘플의 열벡터 표기)이고, 은닉층으로 돌아가는 신호는 `W2`의 연결 강도에 비례한다. 단일 샘플의 1차원 NumPy 배열에서는 `np.outer(a1, delta2)` 등으로 shape를 맞춘다.

> 실수 포인트: 미분하려는 변수에 도달할 때까지 **지나온 모든 중간 함수의 미분**을 곱해야 한다.

---

## 직접 풀어보기

① `f(x)=(3x+1)²`를 미분하라. ② `ŷ=0.8, y=1`에서 출력층 오차항 `δ2`는? ③ `y=0`이면 `δ2`는 어떤 방향으로 작용하는가?

<details>
<summary>정답·힌트 확인</summary>

① `6(3x+1)` ② `-0.2` ③ `δ2=ŷ>0`, 경사하강법은 출력 확률을 낮추는 쪽으로 파라미터를 조정한다.

</details>

## 미션 평가 체크

- [ ] Sigmoid + BCE에서 `δ2=ŷ-y`가 되는 유도를 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 3-4 순전파(Forward Propagation)](STEP_3-4_순전파Forward_Propagation.md) · [전체 목차](../README.md) · [다음: 3-6 역전파(Backpropagation) 직접 구현 →](STEP_3-6_역전파Backpropagation_직접_구현.md)
