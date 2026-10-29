# STEP 5-2. 확률분포·평균·분산

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- 베르누이·정규분포를 직관적으로 이해한다.
- 기댓값과 분산을 계산한다.
- 분포를 그래프로 시각화한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 확률분포는 가능한 값에 확률을 붙인 표

동전의 앞면을 1, 뒷면을 0으로 표시하면 `X~Bernoulli(p)`다.

\[P(X=1)=p,\quad P(X=0)=1-p\]

기댓값(평균)은 `E[X]=p`, 분산은 `Var(X)=p(1-p)`다. 분산은 **평균에서 얼마나 흩어져 있는지** 나타낸다.

일반적으로 이산 확률변수의 평균과 분산은

\[E[X]=\sum_x xP(X=x),\quad Var(X)=E[(X-E[X])^2]\]

이다. 표본의 분산을 계산할 때 `ddof=0`은 모집단식, `ddof=1`은 표본분산 보정식이다.

## 2. 정규분포

종 모양으로 많이 표현되며 평균 `μ`, 표준편차 `σ`가 모양을 정한다.

\[p(x)=\frac1{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\]

연속형 분포에서는 **한 점의 확률**이 아니라 **구간 아래 면적**이 확률이다. 밀도 그래프의 높이가 확률 자체는 아니다.

## 3. NumPy 시각화와 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng=np.random.default_rng(42)
bernoulli=rng.binomial(n=1,p=.7,size=10000)
print('베르누이 평균:',bernoulli.mean(),'이론:',.7)
print('베르누이 분산:',bernoulli.var(),'이론:',.7*.3)

x=np.linspace(-5,5,400)
def normal_pdf(x,mu,sigma):
    return np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

plt.plot(x,normal_pdf(x,0,1),label='mu=0, sigma=1')
plt.plot(x,normal_pdf(x,1,1.5),label='mu=1, sigma=1.5')
plt.title('Normal probability density');plt.xlabel('x');plt.ylabel('density')
plt.legend();plt.grid(alpha=.2);plt.show()
```

## 4. 왜 AI에서 중요한가?

모델은 종종 정답이 특정 확률분포에서 생성되었다고 **가정**하고 파라미터를 학습한다. 예를 들어 관측 오차가 정규분포라고 가정하면 MSE가 자연스럽게 등장한다(5-6).

> 정규분포라고 해서 모든 현실 데이터가 종 모양이라는 뜻은 아니다. 사용하려는 데이터와 가정이 맞는지 확인해야 한다.

---

## 직접 풀어보기

① `Bernoulli(.2)`의 평균과 분산은? ② 표준편차는 분산의 무엇인가? ③ `np.mean`, `np.var`를 같은 샘플에서 계산하라.

<details>
<summary>정답·힌트 확인</summary>

① 평균 `.2`, 분산 `.16`. ② 양의 제곱근이다. ③ 시뮬레이션 비율은 이론값에 근접하지만 정확히 같을 필요는 없다.

</details>

## 미션 평가 체크

- [ ] 확률분포의 기댓값·분산과 정규분포 밀도 그래프를 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-1 확률 기본](STEP_5-1_확률_기본.md) · [전체 목차](../README.md) · [다음: 5-3 가능도(Likelihood) →](STEP_5-3_가능도Likelihood.md)
