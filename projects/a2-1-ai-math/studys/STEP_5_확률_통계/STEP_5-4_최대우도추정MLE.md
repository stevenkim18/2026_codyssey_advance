# STEP 5-4. 최대우도추정(MLE)

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- 가능도를 최대로 하는 파라미터를 추정한다.
- 미분으로 동전 앞면 확률의 MLE를 유도한다.
- 격자 탐색과 경사상승법 결과를 비교한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 가장 그럴듯한 동전 찾기

앞면 7번, 뒷면 3번을 보고 가장 적절한 앞면 확률 `p`를 선택하자. **최대우도추정(Maximum Likelihood Estimation, MLE)**은 가능도 `L(p)`가 최대인 파라미터를 고르는 방법이다.

\[\hat p=\underset{0<p<1}{\operatorname{argmax}}\left[7\log p+3\log(1-p)\right]\]

## 2. 미분으로 풀기

\[\frac{d\ell}{dp}=\frac7p-\frac3{1-p}\]

이를 0에 놓으면 `7(1-p)=3p`이고

\[\boxed{\hat p=\frac7{10}=0.7}\]

이 된다. 일반적으로 독립인 베르누이 관측 `n`번 중 앞면 `k`번이면 `p̂=k/n`이다. 단, 전부 앞면/뒷면이면 경계값 1/0에서 최대에 도달하므로 경계도 포함해 생각해야 한다.

## 3. 격자 탐색과 Gradient Ascent

```python
import numpy as np
import matplotlib.pyplot as plt

k,n=7,10
ps=np.linspace(.001,.999,999)
loglik=k*np.log(ps)+(n-k)*np.log(1-ps)
print('격자 탐색:',ps[np.argmax(loglik)])

# 로그 가능도 최대화: d(ℓ)/dp가 +면 p를 늘린다.
p=.5;lr=.01;trace=[]
for _ in range(80):
    trace.append(p)
    derivative=k/p-(n-k)/(1-p)
    p=np.clip(p+lr*derivative,1e-6,1-1e-6)
print('Gradient ascent 추정:',p)
assert abs(p-.7)<1e-3

plt.plot(ps,loglik,label='log-likelihood')
plt.axvline(.7,color='red',linestyle='--',label='MLE = 0.7')
plt.scatter(trace,[k*np.log(q)+(n-k)*np.log(1-q) for q in trace],
            s=12,alpha=.5,label='ascent steps')
plt.xlabel('p');plt.ylabel('log L(p)');plt.legend();plt.grid(alpha=.2);plt.show()
```

## 4. 경사하강법과의 연결

MLE는 `로그 가능도를 최대화`한다. 같은 문제를 **음의 로그 가능도(NLL)를 최소화**하는 경사하강법으로 바꿀 수도 있다.

\[\operatorname{argmax}\ell(p)=\operatorname{argmin}[-\ell(p)]\]

이 연결 고리가 다음 단원의 BCE, Cross-Entropy를 이해하는 핵심이다.

> 표본 10개만으로 구한 `p̂=.7`에는 불확실성이 있다. MLE는 단 하나의 점 추정치를 제시하며 신뢰구간 자체를 제공하지는 않는다.

---

## 직접 풀어보기

① 20번 중 앞면 13번이면 MLE는? ② `dℓ/dp=0`에서 최적을 구하는 이유는? ③ `-ℓ(p)`에 경사하강법을 적용해 같은 값이 나오는지 확인하라.

<details>
<summary>정답·힌트 확인</summary>

① `13/20=0.65`. ② 내부의 미분 가능한 최대점에서 도함수는 0이기 때문이다(경계는 별도 고려). ③ 음의 로그 가능도는 최소가 `.7`이다.

</details>

## 미션 평가 체크

- [ ] 동전 MLE를 미분, 격자 탐색, 반복 갱신으로 재현할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-3 가능도(Likelihood)](STEP_5-3_가능도Likelihood.md) · [전체 목차](../README.md) · [다음: 5-5 MSE와 Cross-Entropy →](STEP_5-5_MSE와_Cross-Entropy.md)
