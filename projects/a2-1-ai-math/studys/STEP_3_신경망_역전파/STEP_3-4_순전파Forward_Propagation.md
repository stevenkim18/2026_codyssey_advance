# STEP 3-4. 순전파(Forward Propagation)

> 코디세이 A2-1 AI 수학 · STEP 3 신경망과 역전파 · Python/NumPy 입문자용

## 학습 목표

- 2→2→1 신경망의 연산 순서를 안다.
- 중간 결과와 텐서 shape를 출력한다.
- Sigmoid+BCE의 순전파를 직접 구현한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 순전파: 입력부터 예측까지 한 방향으로 계산

`x`가 들어오면 은닉층에서 `z1`, `a1`을 계산하고 출력층에서 `z2`, `ŷ`을 계산한다. 여기서는 역전파 평가 예제와 맞추기 위해 **두 층 모두 Sigmoid**를 쓴다.

\[z_1=xW_1+b_1,\quad a_1=\sigma(z_1)\]
\[z_2=a_1W_2+b_2,\quad \hat y=\sigma(z_2)\]

이진 정답 `y∈{0,1}`에 대해 BCE는

\[L=-[y\log\hat y+(1-y)\log(1-\hat y)]\]

다. 하나의 샘플이므로 평균을 쓰더라도 값이 같다.

## 2. NumPy로 처음부터 실행하기

```python
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

x = np.array([1., .5])                # (2,)
y = 1.0
W1 = np.array([[.2, -.3], [.4, .1]]) # (2,2)
b1 = np.array([.1,-.2])              # (2,)
W2 = np.array([[.7],[-.5]])          # (2,1)
b2 = np.array([.05])                 # (1,)

z1 = x @ W1 + b1                     # (2,)
a1 = sigmoid(z1)                     # (2,)
z2 = a1 @ W2 + b2                    # (1,)
yhat = sigmoid(z2)[0]                # scalar
loss = -(y*np.log(yhat)+(1-y)*np.log(1-yhat))

print('z1:', z1, 'shape:', z1.shape)
print('a1:', a1, 'shape:', a1.shape)
print('z2:', z2, 'shape:', z2.shape)
print('예측값:', yhat, 'BCE:', loss)
assert z1.shape == (2,) and z2.shape == (1,)
assert 0 < yhat < 1 and np.isfinite(loss)
```

## 3. 눈으로 따라가기

은닉층 값 `z1`은 `[0.5,-0.45]`다. Sigmoid를 거쳐 `[0.6225,0.3894]` 근처가 되고, 다시 출력층의 가중합 및 Sigmoid를 통과하면 **약 0.572**라는 예측이 나온다. 정답이 1이므로 손실은 0보다 크다.

## 4. 역전파를 위해 저장할 값

`x`, `z1`, `a1`, `z2`, `yhat`, `y`가 다음 단원의 기울기 계산에 필요하다. 각 값을 **캐시(cache)**로 생각하면 된다.

> 실무 구현에서는 0에 매우 가까운 확률로 로그를 계산할 때 `np.clip`이나 안정적인 logits 기반 BCE를 사용한다.

---

## 직접 풀어보기

① 코드에서 `x=[0,1]`로 바꾸면 예측이 어떻게 달라지는가? ② `W2`가 `(2,1)`이어야 하는 이유는? ③ `a1`을 출력해 Sigmoid 범위인지 확인하라.

<details>
<summary>정답·힌트 확인</summary>

① 입력이 달라지므로 예측과 손실이 달라진다. 직접 값을 출력한다. ② 은닉 뉴런 2개를 최종 출력 뉴런 1개에 연결하기 때문이다. ③ 각 원소가 `(0,1)` 사이여야 한다.

</details>

## 미션 평가 체크

- [ ] 2→2→1의 순전파 중간값, 최종 BCE, 각 텐서 차원을 출력할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 3-3 활성화 함수](STEP_3-3_활성화_함수.md) · [전체 목차](../README.md) · [다음: 3-5 연쇄 법칙(Chain Rule) →](STEP_3-5_연쇄_법칙Chain_Rule.md)
