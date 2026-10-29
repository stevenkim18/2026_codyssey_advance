# STEP 3-6. 역전파(Backpropagation) 직접 구현

> 코디세이 A2-1 AI 수학 · STEP 3 신경망과 역전파 · Python/NumPy 입문자용

## 학습 목표

- 2→2→1 Sigmoid+BCE의 순전파와 역전파를 직접 연결한다.
- 각 파라미터의 기울기를 NumPy로 계산한다.
- Gradient Checking으로 계산을 검증한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 전체 흐름 한눈에 보기

**순전파**는 `x→z1→a1→z2→ŷ→L`로 진행한다. **역전파**는 손실에서 시작해 `W2,b2`의 기울기를 구하고, 은닉층으로 전달하여 `W1,b1`의 기울기를 구한다.

단일 샘플 기준 공식:

\[\delta_2=\hat y-y\]
\[\frac{\partial L}{\partial W_2}=a_1\,\delta_2,\quad \frac{\partial L}{\partial b_2}=\delta_2\]
\[\delta_1=(W_2\,\delta_2)\odot a_1\odot(1-a_1)\]
\[\frac{\partial L}{\partial W_1}=x^T\delta_1,\quad\frac{\partial L}{\partial b_1}=\delta_1\]

표기는 각 벡터의 행/열 약속에 따라 달라질 수 있다. 아래 코드에서는 `x, a1, delta1`이 1차원 배열이고, 외적은 `np.outer`로 계산한다.

## 2. 한 파일에서 바로 실행되는 구현

```python
import numpy as np

x = np.array([1., .5]); y = 1.0
W1 = np.array([[.2,-.3],[.4,.1]])
b1 = np.array([.1,-.2])
W2 = np.array([[.7],[-.5]])
b2 = np.array([.05])

def sigmoid(z): return 1/(1+np.exp(-z))

def forward(W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    yhat = sigmoid(z2)[0]
    p = np.clip(yhat,1e-12,1-1e-12)
    L = -(y*np.log(p)+(1-y)*np.log(1-p))
    return L,(z1,a1,z2,yhat)

L, (z1,a1,z2,yhat) = forward(W1,b1,W2,b2)
delta2 = yhat - y             # scalar

dW2 = a1[:,None] * delta2    # (2,1)
db2 = np.array([delta2])     # (1,)
delta1 = (W2[:,0]*delta2)*a1*(1-a1) # (2,)
dW1 = np.outer(x, delta1)    # (2,2)
db1 = delta1                # (2,)

print('ŷ=', yhat, 'BCE=',L)
for name,value in [('dW1',dW1),('db1',db1),('dW2',dW2),('db2',db2)]:
    print(name, value.shape, '\n',value)

# Gradient checking: 각 파라미터를 하나씩 ±h 흔들어서 확인
h=1e-5
for arr, grad, name in [(W1,dW1,'W1'),(b1,db1,'b1'),(W2,dW2,'W2'),(b2,db2,'b2')]:
    estimated=np.zeros_like(arr)
    for index in np.ndindex(arr.shape):
        old=arr[index]
        arr[index]=old+h; plus=forward(W1,b1,W2,b2)[0]
        arr[index]=old-h; minus=forward(W1,b1,W2,b2)[0]
        arr[index]=old
        estimated[index]=(plus-minus)/(2*h)
    error=np.max(np.abs(grad-estimated))
    print(f'{name} max error = {error:.2e}')
    assert error < 1e-4

# 가중치 1회 갱신 후 손실이 줄어드는지도 확인한다.
learning_rate=.1
new_L,_=forward(W1-learning_rate*dW1, b1-learning_rate*db1,
                W2-learning_rate*dW2, b2-learning_rate*db2)
print('업데이트 전:',L,'후:',new_L)
assert new_L < L
```

## 3. 코드가 계산한 것을 말로 설명하기

1. `delta2`는 **정답과 예측의 차이**다.
2. `dW2`는 은닉층 출력이 출력층 손실에 얼마나 기여했는지 나타낸다.
3. `delta1`은 출력층의 오차를 은닉층에 전달하면서 Sigmoid 도함수를 곱한 것이다.
4. `dW1`은 입력 `x`와 은닉층 오차 `delta1`의 외적이다.
5. 중앙차분 결과와 기울기를 비교해 부호와 차원을 검증한다.

> `np.clip`은 이번 예제에서 극단적인 `log(0)`을 방지하기 위해 사용한다. 학습 값이 정확히 0/1로 잘리는 지점에서는 해석적 미분과 수치미분의 비교가 까다로워질 수 있다.

---

## 직접 풀어보기

① `db2`가 왜 출력 뉴런 1개에 대응하는 모양 `(1,)`인지 설명하라. ② `learning_rate=.01`로 바꾸어 손실 감소량을 비교하라. ③ `W1[0,0]`의 해석적 기울기와 수치미분값을 직접 비교하라.

<details>
<summary>정답·힌트 확인</summary>

① 출력층 편향은 한 뉴런에 하나씩 있다. ② 대체로 더 작은 이동을 하므로 한 번의 손실 변화도 작다. ③ 마지막 검증 루프의 `W1` 행렬 `[0,0]` 성분을 출력하면 된다.

</details>

## 미션 평가 체크

- [ ] 순전파→BCE→역전파→1회 업데이트와 각 파라미터 Gradient 오차 `1e-4` 검증을 재현할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 3-5 연쇄 법칙(Chain Rule)](STEP_3-5_연쇄_법칙Chain_Rule.md) · [전체 목차](../README.md) · [다음: 4-1 손실 함수(Loss Function) →](../STEP_4_최적화/STEP_4-1_손실_함수Loss_Function.md)
