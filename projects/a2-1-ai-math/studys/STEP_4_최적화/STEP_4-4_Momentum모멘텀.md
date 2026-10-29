# STEP 4-4. Momentum(모멘텀)

> 코디세이 A2-1 AI 수학 · STEP 4 최적화 · Python/NumPy 입문자용

## 학습 목표

- 기존 GD와 이동 속도 누적의 차이를 이해한다.
- Momentum 업데이트를 NumPy로 직접 구현한다.
- 동일한 초기값에서 두 방법을 비교한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 눈덩이와 관성의 비유

기본 경사하강법은 그 순간의 기울기만 보고 한 걸음 움직인다. Momentum은 이전 걸음의 방향을 일부 기억한다. 한 방향으로 계속 가면 속도가 붙고, 지그재그 이동은 누그러질 수 있다.

## 2. 수식

\[v_t=\beta v_{t-1}+\nabla L(w_t)\]
\[w_{t+1}=w_t-\alpha v_t\]

`β`는 이전 이동 정보를 얼마나 유지할지 정하는 계수다. 흔히 `0.9` 정도로 시작하지만 데이터/학습률에 따라 조정한다. 이 표현은 기울기를 속도에 누적하는 관례이며, 문헌에 따라 `v`의 부호/학습률 배치가 다른 동등한 표기가 있다.

## 3. Vanilla GD와 Momentum 비교

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1.,2.,3.]); y=2*x

def loss(w): return np.mean((w*x-y)**2)
def grad(w): return np.mean(2*(w*x-y)*x)

def train(momentum=False, lr=.03, beta=.9, epochs=60):
    w=0.; v=0.; losses=[]; weights=[]
    for _ in range(epochs):
        losses.append(loss(w)); weights.append(w)
        g=grad(w)
        if momentum:
            v=beta*v+g
            w-=lr*v
        else:
            w-=lr*g
    return np.array(weights),np.array(losses)

fig,axes=plt.subplots(1,2,figsize=(11,4))
for is_momentum,name in [(False,'Vanilla GD'),(True,'Momentum')]:
    w,ls=train(is_momentum)
    print(name,'last w=',w[-1],'last logged loss=',ls[-1])
    axes[0].plot(w,label=name)
    axes[1].semilogy(ls+1e-16,label=name)
axes[0].axhline(2,color='gray',ls='--')
axes[0].set(xlabel='epoch',ylabel='w',title='Weight')
axes[1].set(xlabel='epoch',ylabel='MSE',title='Loss')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
```

## 4. 무엇을 비교해야 하나?

두 알고리즘의 출발값과 데이터, 학습 횟수를 같게 놓는다. 손실 감소 속도, 오버슈트(최적값을 넘어가는 현상), 최종 손실을 본다. **Momentum이 모든 문제에서 항상 더 빠르거나 안정적인 것은 아니다.** 특히 학습률이 크면 관성 때문에 크게 흔들릴 수 있다.

> 혼동 주의: Momentum은 Gradient를 구하는 새로운 방식이 아니라, 구해진 Gradient를 **업데이트에 반영하는 방식**이다.

---

## 직접 풀어보기

① `β=0`이면 Momentum 업데이트는 무엇과 같아지는가? ② `β=.5`, `.9`, `.99`를 실험하라. ③ 동일한 학습률에서 초기 손실 감소가 빠르더라도 최종 손실을 비교해야 하는 이유는?

<details>
<summary>정답·힌트 확인</summary>

① Vanilla GD가 된다. ② 이전 이동의 누적량이 달라져 오버슈트/진동도 달라진다. ③ 초기 감소 속도만으로 최종 수렴 여부와 안정성은 알 수 없기 때문이다.

</details>

## 미션 평가 체크

- [ ] Momentum과 Vanilla GD의 손실 곡선을 동일한 Figure에서 비교하고 차이를 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 4-3 학습률(Learning Rate)](STEP_4-3_학습률Learning_Rate.md) · [전체 목차](../README.md) · [다음: 4-5 학습 과정 시각화 및 비교 →](STEP_4-5_학습_과정_시각화_및_비교.md)
