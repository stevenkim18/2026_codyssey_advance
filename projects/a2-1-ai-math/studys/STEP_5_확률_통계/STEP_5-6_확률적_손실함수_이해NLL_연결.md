# STEP 5-6. 확률적 손실함수 이해(NLL 연결)

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- 확률모형에서 음의 로그 가능도(NLL)가 나오는 과정을 이해한다.
- 정규분포-MSE, 베르누이-BCE, 카테고리-CE를 연결한다.
- 상수항과 스케일 계수의 의미를 구분한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. MLE와 손실 함수는 서로 다른 이야기일까?

관측한 정답이 어떤 확률분포에서 생성되었다고 가정하고, 데이터를 가장 잘 설명하는 파라미터를 찾는 것이 MLE다. 확률의 곱이 불편하므로 로그를 취하고, 최대화를 최소화로 바꾸면 **음의 로그 가능도(Negative Log-Likelihood, NLL)**가 된다.

\[\arg\max_\theta P(D\mid\theta)=\arg\min_\theta[-\log P(D\mid\theta)]\]

## 2. 정규분포를 가정하면 MSE

정답 `y_i`가 `N(ŷ_i, σ²)`에서 생성되고 `σ²`가 모든 샘플에서 일정한 고정값이라면

\[-\log P(D\mid\theta)=\frac{1}{2\sigma^2}\sum_i(y_i-\hat y_i)^2+\text{상수}\]

이다. 따라서 **고정된 분산에서** NLL 최소화와 SSE/MSE 최소화는 같은 최적 예측을 낸다. 분산까지 함께 학습한다면 추가 항이 있으므로 단순 MSE와 완전히 같다고 할 수 없다.

## 3. 베르누이 분포면 BCE

정답 `y∈{0,1}`, 예측 확률 `p`일 때

\[P(y\mid p)=p^y(1-p)^{1-y}\]
\[-\log P(y\mid p)=-[y\log p+(1-y)\log(1-p)]\]

바로 BCE 한 샘플 값이다.

## 4. 카테고리 분포면 CE

서로 배타적인 `C`개 클래스 중 하나가 정답이라면, 정답 클래스를 원-핫 `y_c`로 나타내어

\[-\log P(y\mid \mathbf p)=-\sum_c y_c\log p_c\]

를 얻는다. 이것이 다중 클래스 CE다.

## 5. 실제 값으로 비교

```python
import numpy as np

# 정규모형: sigma=1, 관측 y=2, 예측 mean=1.5
sigma=1.;y=2.;pred=1.5
normal_nll=0.5*np.log(2*np.pi*sigma**2)+(y-pred)**2/(2*sigma**2)
print('정규 NLL:',normal_nll)
print('오차제곱×계수:',(y-pred)**2/(2*sigma**2))
print('차이는 정규화 상수:',.5*np.log(2*np.pi*sigma**2))

# 베르누이: 정답 1, 예측 p=.8
p=.8
bernoulli_nll=-np.log(p)
print('베르누이 NLL = BCE:',bernoulli_nll)

# 카테고리: 정답 2번 클래스, 예측확률 [.1,.2,.7]
prob=np.array([.1,.2,.7]); target=np.array([0,0,1])
cat_nll=-np.sum(target*np.log(prob))
print('카테고리 NLL = CE:',cat_nll)
```

## 6. 세 손실을 한 문장으로 정리

- **MSE**: 정규 오차모형에서 유도되는 제곱오차 기반 손실.
- **BCE**: 베르누이 확률모형의 NLL.
- **CE**: 카테고리 확률모형의 NLL.

> 모델의 확률 가정에 따라 손실의 해석이 달라진다. 손실 함수를 단지 공식을 외워서 쓰지 말고 **데이터의 종류와 모델의 출력**을 함께 확인하자.

---

## 직접 풀어보기

① 정규분포의 분산을 더 크게 고정하면 제곱오차 항의 계수는 어떻게 바뀌는가? ② `y=0,p=.9`의 BCE는? ③ 이진 분류와 다중 분류에서 어떤 확률모형을 가정했는지 써 보라.

<details>
<summary>정답·힌트 확인</summary>

① `1/(2σ²)`가 작아진다. ② `-log(.1)≈2.3026`. ③ 각각 베르누이와 카테고리 분포.

</details>

## 미션 평가 체크

- [ ] MSE/BCE/CE가 NLL과 연결되는 과정을 관측 가정까지 포함해 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-5 MSE와 Cross-Entropy](STEP_5-5_MSE와_Cross-Entropy.md) · [전체 목차](../README.md) · [다음: 6-1 선형대수 실습 정리 →](../STEP_6_통합_제출/STEP_6-1_선형대수_실습_정리.md)
