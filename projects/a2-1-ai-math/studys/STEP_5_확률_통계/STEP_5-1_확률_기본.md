# STEP 5-1. 확률 기본

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- 사건·표본공간·조건부 확률을 구분한다.
- 동전 던지기를 확률 모델로 표현한다.
- 기본 확률을 NumPy 시뮬레이션으로 확인한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 확률은 일어나기 전 불확실성을 숫자로 표현한다

공정한 동전을 던질 때 앞면이 나올 확률은 0.5다. 표본공간은 `{앞면, 뒷면}`이고, 앞면이 나오는 것을 **사건**이라고 한다. 확률은 `0≤P(A)≤1`을 만족한다.

## 2. 덧셈·곱셈·조건부 확률

- 서로 배타적인 사건 `A,B`: `P(A 또는 B)=P(A)+P(B)`
- 독립적인 사건 `A,B`: `P(A 그리고 B)=P(A)P(B)`
- 조건부 확률: `P(A|B)=P(A 그리고 B)/P(B)` (단, `P(B)>0`)

**독립**과 **배타**는 다르다. 독립은 하나의 결과가 다른 결과의 확률을 바꾸지 않는다는 뜻이고, 배타는 동시에 일어날 수 없다는 뜻이다.

## 3. NumPy 동전 던지기

```python
import numpy as np
import matplotlib.pyplot as plt

rng=np.random.default_rng(42)
heads=(rng.random(1000)<.5).astype(int)
ratio=np.cumsum(heads)/np.arange(1,len(heads)+1)
print('1000번 실험 앞면 비율:',heads.mean())
plt.plot(np.arange(1,1001),ratio,label='Observed P(heads)')
plt.axhline(.5,color='red',linestyle='--',label='True P=.5')
plt.xlabel('Number of trials');plt.ylabel('Cumulative frequency')
plt.ylim(0,1);plt.legend();plt.grid(alpha=.2);plt.show()
```

## 4. AI와의 연결

분류 모델은 특정 클래스의 확률을 출력한다. 이것이 반드시 **현실에서 완벽히 보정된 확률**이라는 보장은 없다. 학습 과정에서 확률 예측을 정답에 가깝게 조정하도록 BCE, CE 같은 손실을 사용한다.

> 적은 횟수의 실험에서는 앞면 비율이 0.5에서 멀어질 수 있다. 표본 수가 증가할수록 장기 평균이 0.5에 가까워지는 것이 대수의 법칙의 핵심 직관이다.

---

## 직접 풀어보기

① 동전을 두 번 던져 둘 다 앞면일 확률은? ② 두 번 중 적어도 한 번 앞면은? ③ 공정한 동전을 10번 던졌는데 앞면이 8번 나왔다면 불공정하다고 단정할 수 있는가?

<details>
<summary>정답·힌트 확인</summary>

① `0.5×0.5=0.25` ② `1-P(두 번 다 뒷면)=0.75` ③ 아니다. 충분한 증거와 통계적 검정 없이 단정할 수 없다.

</details>

## 미션 평가 체크

- [ ] 확률과 실험 빈도의 차이, 독립성의 의미를 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 4-5 학습 과정 시각화 및 비교](../STEP_4_최적화/STEP_4-5_학습_과정_시각화_및_비교.md) · [전체 목차](../README.md) · [다음: 5-2 확률분포·평균·분산 →](STEP_5-2_확률분포·평균·분산.md)
