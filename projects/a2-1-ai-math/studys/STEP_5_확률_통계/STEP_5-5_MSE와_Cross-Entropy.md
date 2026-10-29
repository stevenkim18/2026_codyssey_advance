# STEP 5-5. MSE와 Cross-Entropy

> 코디세이 A2-1 AI 수학 · STEP 5 확률과 통계 · Python/NumPy 입문자용

## 학습 목표

- MSE, BCE, Categorical CE를 구분한다.
- Sigmoid·Softmax와 손실의 관계를 이해한다.
- 손실 곡선과 NumPy 구현을 비교한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 예측에 맞는 손실을 사용한다

**MSE**는 연속적인 값(가격, 온도 등)의 예측 오차를 제곱한다. **BCE**는 이진 분류의 예측 확률과 정답을 비교한다. **Categorical Cross-Entropy(다중 클래스 CE)**는 여러 클래스 중 정답 클래스에 부여한 확률을 평가한다.

\[\operatorname{MSE}=\frac1n\sum_i(\hat y_i-y_i)^2\]

\[\operatorname{BCE}=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]\]

\[\operatorname{CE}=-\sum_{c=1}^{C}y_c\log p_c\]

마지막 식은 **샘플 1개 기준**이며 다수 샘플에서는 평균을 취할 수 있다. 정답이 원-핫 벡터라면 정답 클래스의 `-log(p)`만 남는다.

## 2. 두 가지 활성화 함수를 기억하자

- `Sigmoid`: 출력 하나를 `(0,1)` 구간에 매핑해 이진 분류 확률로 사용.
- `Softmax`: 여러 클래스 점수(`logits`)를 합이 1인 확률 벡터로 사용.

## 3. NumPy 계산 및 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):return 1/(1+np.exp(-z))
def softmax(logits):
    logits=np.asarray(logits,dtype=float)
    exps=np.exp(logits-np.max(logits))
    return exps/exps.sum()
def mse(y,p):return np.mean((np.asarray(y)-np.asarray(p))**2)
def bce(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    y=np.asarray(y,dtype=float)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def ce(onehot,probs):
    probs=np.clip(probs,1e-12,1.)
    return -np.sum(onehot*np.log(probs))

print('MSE:',mse([2.,4.],[2.5,3.]))
print('BCE:',bce([1,0],[.9,.2]))
probs=softmax([2.,1.,.1])
print('Softmax:',probs,'CE(class0):',ce(np.array([1.,0.,0.]),probs))

p=np.linspace(.001,.999,300)
plt.plot(p,(1-p)**2,label='Squared error (y=1)')
plt.plot(p,-np.log(p),label='BCE (y=1)')
plt.ylim(0,6);plt.xlabel('Predicted probability p')
plt.ylabel('Loss');plt.title('Same target, different penalties')
plt.grid(alpha=.2);plt.legend();plt.show()
```

## 4. 그래프 해석

모두 `p=1`에서 최소가 되지만, `p→0`에서는 BCE가 훨씬 크게 증가한다. MSE와 BCE는 같은 상황에 항상 바꿔 쓰는 손실이 아니다. 모델 출력의 확률 해석과 **데이터 생성 가정**을 보고 결정한다.

> 실제 분류 코드에서는 `logits`를 받아 안정적으로 BCE/CE를 계산하는 구현이 수치적으로 더 안전하다.

---

## 직접 풀어보기

① `y=1,p=.1`일 때 제곱오차와 BCE를 각각 계산하라. ② Softmax 출력 확률 합은? ③ 원-핫 정답 `[0,1,0]`의 CE를 적어 보라.

<details>
<summary>정답·힌트 확인</summary>

① `(1-.1)²=.81`, `-log(.1)≈2.3026`. ② `1`. ③ `-log(p₁)`(두 번째 클래스 확률).

</details>

## 미션 평가 체크

- [ ] MSE/BCE/CE 손실 곡선과 Sigmoid/Softmax 역할을 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-4 최대우도추정(MLE)](STEP_5-4_최대우도추정MLE.md) · [전체 목차](../README.md) · [다음: 5-6 확률적 손실함수 이해(NLL 연결) →](STEP_5-6_확률적_손실함수_이해NLL_연결.md)
