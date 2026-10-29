# STEP 4-2. 경사하강법(Gradient Descent)

> 코디세이 A2-1 AI 수학 · STEP 4 최적화 · Python/NumPy 입문자용

## 학습 목표

- 손실의 기울기 반대 방향으로 파라미터를 이동시킨다.
- 선형회귀 예제로 MSE 기울기를 직접 구현한다.
- 손실이 반복해서 감소하는지 검증한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 산 아래로 내려가는 규칙

안개가 낀 산에서 내리막 방향만 알 수 있다면 한 걸음씩 내려갈 수 있다. 경사하강법은 손실 함수에서 같은 일을 한다.

\[w_{t+1}=w_t-\alpha\frac{\partial L}{\partial w}\]

`α`는 한 걸음의 크기인 학습률이다. 기울기가 양수면 `w`를 줄이고, 음수면 늘린다.

## 2. 선형회귀로 손계산

`ŷ=wx`, `x=[1,2,3]`, `y=[2,4,6]`이라면

\[L(w)=\frac13\sum_{i=1}^{3}(wx_i-y_i)^2\]

\[\frac{dL}{dw}=\frac23\sum_i(wx_i-y_i)x_i\]

최적값은 `w=2`다. `w=0`에서 기울기는 `-28/3×2=-56/3`이므로 한 걸음 이동하면 `w`가 증가한다.

## 3. NumPy 직접 구현

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1.,2.,3.]); y=np.array([2.,4.,6.])
w=0.0; lr=.1; history=[]; weights=[]
for epoch in range(40):
    pred=w*x
    loss=np.mean((pred-y)**2)
    grad=np.mean(2*(pred-y)*x)
    history.append(loss); weights.append(w)
    w -= lr*grad
print('최종 w=',w,'최종 손실=',np.mean((w*x-y)**2))
assert abs(w-2)<1e-3

fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].plot(history); ax[0].set(title='Loss by epoch',xlabel='epoch',ylabel='MSE')
ax[1].plot(weights); ax[1].axhline(2,color='gray',ls='--')
ax[1].set(title='Weight approaches 2',xlabel='epoch',ylabel='w')
plt.tight_layout(); plt.show()
```

## 4. 부호 검증

`w=0`이라면 `pred=0`, 정답은 양수이므로 `(pred-y)`가 음수, 기울기도 음수다. 업데이트 식에는 마이너스 부호가 있으므로 `w`가 커진다. **이 부호 관계를 말로 설명**할 수 있어야 한다.

> 여러 가중치가 있다면 `w` 하나 대신 벡터 `W`에 Gradient를 적용한다.

---

## 직접 풀어보기

① `L(w)=(w-2)²`에서 `w=0`, `α=.1`이면 새 `w`는? ② 위 코드의 `w`를 5로 시작해 보라. ③ 학습률을 0으로 설정하면 무엇이 달라지는가?

<details>
<summary>정답·힌트 확인</summary>

① `dL/dw=-4`이므로 `w=0.4`. ② 기울기의 부호가 반대로 시작하지만 적절한 학습률이면 2로 간다. ③ 가중치가 갱신되지 않는다.

</details>

## 미션 평가 체크

- [ ] 손실 곡선과 가중치 업데이트 과정을 그래프로 제시할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 4-1 손실 함수(Loss Function)](STEP_4-1_손실_함수Loss_Function.md) · [전체 목차](../README.md) · [다음: 4-3 학습률(Learning Rate) →](STEP_4-3_학습률Learning_Rate.md)
