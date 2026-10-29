# STEP 4-3. 학습률(Learning Rate)

> 코디세이 A2-1 AI 수학 · STEP 4 최적화 · Python/NumPy 입문자용

## 학습 목표

- 학습률이 너무 작거나 크면 생기는 문제를 구분한다.
- 수렴·진동·발산 그래프를 직접 만든다.
- 이 예제의 수렴 조건을 수학으로 유도한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 계단을 내려갈 때의 걸음 크기

아주 작은 걸음은 안전하지만 느리다. 적당한 걸음은 빠르다. 너무 큰 걸음은 골짜기를 건너뛰어 손실이 커질 수 있다. 학습률 `α`는 이런 걸음 크기다.

## 2. 이전 선형회귀 식에서 수렴 조건 유도

`x=[1,2,3]`, `y=2x`라면

\[L(w)=\frac{14}{3}(w-2)^2\]

\[\frac{dL}{dw}=\frac{28}{3}(w-2)\]

업데이트하면

\[w_{t+1}-2=\left(1-\frac{28}{3}\alpha\right)(w_t-2)\]

따라서 오차의 크기가 반복해서 줄어들려면

\[\left|1-\frac{28}{3}\alpha\right|<1\Longleftrightarrow 0<\alpha<\frac{3}{14}\approx0.2143\]

범위 밖이라고 항상 한두 번 만에 발산하는 것은 아니지만, 이 이차 손실의 단순 GD에서는 위 범위가 수렴 조건이다.

## 3. 수렴·진동·발산 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1.,2.,3.]); y=2*x

def train(lr, epochs=25):
    w=0.; losses=[]; weights=[]
    for _ in range(epochs):
        losses.append(np.mean((w*x-y)**2))
        weights.append(w)
        grad=np.mean(2*(w*x-y)*x)
        w-=lr*grad
    return np.asarray(weights),np.asarray(losses)

fig,axes=plt.subplots(1,2,figsize=(11,4))
for lr in [.01,.1,.25]:
    ws,ls=train(lr)
    axes[0].plot(ws,label=f'lr={lr}')
    axes[1].semilogy(ls+1e-16,label=f'lr={lr}')
axes[0].axhline(2,color='gray',ls='--')
axes[0].set(xlabel='epoch',ylabel='w',title='Weight trajectory')
axes[1].set(xlabel='epoch',ylabel='MSE (log scale)',title='Convergence vs divergence')
for ax in axes: ax.legend();ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
print('수렴 상한:',3/14)
```

## 4. 관찰 결과를 설명해 보자

- `lr=.01`: 천천히 2에 가까워진다.
- `lr=.1`: 비교적 빠르게 수렴한다.
- `lr=.25`: 이 예제의 안정 조건을 넘어서 오차가 커지며 발산한다.

> 실제 딥러닝 손실은 이처럼 단순한 포물선이 아니며, 안정 학습률을 보편적인 한 숫자로 정할 수는 없다.

---

## 직접 풀어보기

① `α=.2`는 이 예제에서 수렴하는가? ② `α=.3`이면? ③ `lr`를 바꾼 그래프에서 발산을 설명할 증거 2개를 말해 보라.

<details>
<summary>정답·힌트 확인</summary>

① `0.2 < 3/14`여서 수렴(진동하며 느릴 수 있음). ② 발산. ③ `w`가 2에서 멀어짐, MSE가 증가함.

</details>

## 미션 평가 체크

- [ ] 학습률별 수렴·진동·발산 그래프와 최종 손실을 비교할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 4-2 경사하강법(Gradient Descent)](STEP_4-2_경사하강법Gradient_Descent.md) · [전체 목차](../README.md) · [다음: 4-4 Momentum(모멘텀) →](STEP_4-4_Momentum모멘텀.md)
