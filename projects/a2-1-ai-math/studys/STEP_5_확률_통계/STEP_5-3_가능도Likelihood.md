# STEP 5-3. 가능도(Likelihood)

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- 확률과 가능도의 관점을 구분한다.
- 동전 던지기 데이터로 가능도 함수를 만든다.
- 가능도와 로그 가능도 그래프를 비교한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 관점의 차이: 무엇을 고정하는가?

**확률(Probability)**: 동전 앞면 확률 `p`를 알고 있을 때, 어떤 관측 결과가 나올까?

**가능도(Likelihood)**: 던져서 얻은 결과를 이미 알고 있을 때, 어떤 `p`가 이 데이터를 가장 그럴듯하게 설명할까?

수식은 같은 확률모형을 쓰지만 **데이터를 고정하고 파라미터 p를 움직여 비교**하는 것이 가능도다. 가능도는 파라미터에 대한 확률분포가 아니므로 p에 대해 적분해서 1이 될 필요는 없다.

## 2. 동전 10번: 앞면 7번, 뒷면 3번

개별 관측의 순서까지 고정하면

\[L(p)=p^7(1-p)^3\]

이다. 앞면 개수만 관측했다면 `10C7`이라는 상수 계수가 추가되지만, **최적의 p에는 영향이 없다**.

로그를 취하면 곱셈이 덧셈으로 바뀐다.

\[\ell(p)=\log L(p)=7\log p+3\log(1-p)\]

로그는 증가 함수이므로 `L(p)`가 최대인 위치와 `ℓ(p)`가 최대인 위치가 같다.

## 3. 가능도 그래프 그리기

```python
import numpy as np
import matplotlib.pyplot as plt

heads,tails=7,3
p=np.linspace(.001,.999,999)
likelihood=p**heads*(1-p)**tails
log_likelihood=heads*np.log(p)+tails*np.log(1-p)

best=p[np.argmax(likelihood)]
print('최대 가능도 근처의 p:',best)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(p,likelihood,label='Likelihood')
axes[1].plot(p,log_likelihood,label='Log-likelihood')
for ax in axes:
    ax.axvline(.7,color='red',ls='--',label='p=0.7')
    ax.set_xlabel('p');ax.legend();ax.grid(alpha=.2)
axes[0].set_ylabel('L(p)');axes[1].set_ylabel('log L(p)')
plt.tight_layout();plt.show()
```

## 4. 로그 가능도를 쓰는 이유

관측 수가 커지면 작은 확률을 계속 곱하면서 부동소수점 언더플로가 발생할 수 있다. 로그를 쓰면 곱셈이 합으로 바뀌어 수치적으로 다루기 쉽다. `p=0` 또는 `1`에서는 `log(0)`에 주의해야 하므로 위 코드에서는 양끝을 피했다.

> 가능도의 숫자가 높다고 그 파라미터가 진짜일 확률이 높다는 뜻은 아니다. 이를 파라미터의 사후확률로 해석하려면 베이즈 규칙과 사전분포가 더 필요하다.

---

## 직접 풀어보기

① 5번 중 앞면 4번이면 `L(p)`는? ② `p=.5`와 `.8`의 가능도를 비교하라. ③ 로그를 취해도 최대가 같은 이유는?

<details>
<summary>정답·힌트 확인</summary>

① `p⁴(1-p)`. ② `.8⁴×.2`와 `.5⁵`를 비교하면 `.8`이 더 크다. ③ `log`가 엄격히 증가하는 함수이기 때문이다.

</details>

## 미션 평가 체크

- [ ] Probability와 Likelihood 차이를 고정하는 대상의 관점에서 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-2 확률분포·평균·분산](STEP_5-2_확률분포·평균·분산.md) · [전체 목차](../README.md) · [다음: 5-4 최대우도추정(MLE) →](STEP_5-4_최대우도추정MLE.md)
