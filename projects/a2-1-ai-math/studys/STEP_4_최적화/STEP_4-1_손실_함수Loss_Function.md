# STEP 4-1. 손실 함수(Loss Function)

> 코디세이 A2-1 AI 수학 · STEP 4 최적화 · Python/NumPy 입문자용

## 학습 목표

- 손실이 모델의 틀린 정도를 수치화한다는 의미를 안다.
- MSE와 BCE가 적합한 문제 유형을 구분한다.
- 손실 그래프를 NumPy로 그린다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 목표 지점까지의 거리: 손실

AI 모델은 정답을 맞히도록 가중치를 바꾼다. 그런데 정답에 얼마나 가까운지 **하나의 숫자**로 평가해야 업데이트 방향을 정할 수 있다. 이 숫자가 손실(loss)이다. 목표는 학습 데이터에서 손실을 작게 만드는 것이다.

## 2. 회귀 문제의 MSE

\[\operatorname{MSE}=\frac1n\sum_{i=1}^{n}(\hat y_i-y_i)^2\]

정답이 `y=[2,4,6]`, 예측이 `ŷ=[1,5,7]`이면 오차 제곱이 `[1,1,1]`이므로 MSE는 1이다. 제곱을 사용하므로 큰 오차에 더 민감하다.

## 3. 이진 분류의 BCE

\[\operatorname{BCE}=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]\]

`y=1`인데 모델이 `p=0.9`라고 말하면 손실은 작지만 `p=0.01`이면 매우 크다. **확신에 찬 오답을 크게 벌주는** 성질이 있다.

## 4. 코드와 그래프

```python
import numpy as np
import matplotlib.pyplot as plt

def mse(y,pred): return np.mean((np.asarray(y)-np.asarray(pred))**2)
def bce(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    y=np.asarray(y,dtype=float)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

print('MSE=',mse([2,4,6],[1,5,7]))
print('BCE(y=1,p=.9)=',bce([1],[.9]))
print('BCE(y=1,p=.01)=',bce([1],[.01]))

w=np.linspace(-2,6,300)
prob=np.linspace(.001,.999,300)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(w,(w-2)**2)
axes[0].set(xlabel='w',ylabel='MSE-like loss',title='Minimum at w=2')
axes[1].plot(prob,-np.log(prob),label='y=1')
axes[1].plot(prob,-np.log(1-prob),label='y=0')
axes[1].set(xlabel='predicted p',ylabel='BCE',title='Binary cross-entropy')
axes[1].legend()
for ax in axes: ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
```

## 5. 손실이 작다고 성능이 충분할까?

학습 데이터의 손실만 지나치게 작게 만들면 새로운 데이터에 약해지는 과적합이 발생할 수 있다. **학습용/검증용 데이터**를 나누는 이유다. 정확도는 정답 개수 비율이고, 손실은 예측의 확신 정도까지 반영할 수 있으므로 서로 다른 지표다.

---

## 직접 풀어보기

① `y=[0,2]`, `ŷ=[1,4]`의 MSE는? ② `y=1`일 때 `p=.5`와 `p=.99` 중 손실이 작은 것은? ③ 확신한 오답의 BCE가 큰 이유를 설명하라.

<details>
<summary>정답·힌트 확인</summary>

① `(1+4)/2=2.5` ② `.99` ③ 정답이 1인데 확률이 0에 가깝다면 `-log(p)`가 매우 커진다.

</details>

## 미션 평가 체크

- [ ] MSE와 BCE 그래프를 그리고 두 손실의 적용 상황을 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 3-6 역전파(Backpropagation) 직접 구현](../STEP_3_신경망_역전파/STEP_3-6_역전파Backpropagation_직접_구현.md) · [전체 목차](../README.md) · [다음: 4-2 경사하강법(Gradient Descent) →](STEP_4-2_경사하강법Gradient_Descent.md)
