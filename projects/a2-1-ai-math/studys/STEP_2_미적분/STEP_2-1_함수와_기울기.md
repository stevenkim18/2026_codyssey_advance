# STEP 2-1. 함수와 기울기

> 코디세이 A2-1 AI 수학 · STEP 2 미적분 · Python/NumPy 입문자용

## 학습 목표

- 함수의 입력과 출력, 그래프의 의미를 말할 수 있다.
- 직선의 기울기를 두 점으로 계산한다.
- 기울기가 AI 학습에서 갖는 의미를 연결한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 직관: 함수는 숫자를 넣으면 결과가 나오는 자판기

자판기에 2를 넣으면 5가 나오고, 3을 넣으면 7이 나온다면 `f(x)=2x+1`이라는 규칙이 있을 수 있다. 여기에서 `x`는 입력, `f(x)`는 출력이다.

\[f(x)=2x+1,\qquad f(2)=5,\quad f(3)=7\]

**기울기**는 입력을 1만큼 바꿀 때 출력이 얼마나 변하는지를 뜻한다. 두 점 `(x₁,y₁)`, `(x₂,y₂)`가 있을 때 평균 기울기는 다음과 같다.

\[\text{기울기}=\frac{y_2-y_1}{x_2-x_1}\]

위 함수에서는 `(3-2)`만큼 움직일 때 출력이 `(7-5)`만큼 변하므로 기울기가 2다. 직선은 어느 구간에서 계산해도 같은 기울기를 갖는다.

## 2. 굽은 곡선의 기울기는 다르다

`f(x)=x²`라면 `(1,1)`과 `(2,4)` 사이 기울기는 3, `(2,4)`와 `(3,9)` 사이는 5다. 곡선에서는 **지점마다 기울기가 다를 수 있다**. 이를 정확히 다루는 것이 다음 단원의 미분이다.

## 3. NumPy와 Matplotlib로 그려 보기

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
linear = 2*x + 1
curve = x**2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(x, linear, label='f(x)=2x+1')
axes[0].scatter([2, 3], [5, 7], color='black')
axes[0].set_title('Constant slope = 2')
axes[1].plot(x, curve, label='f(x)=x^2')
axes[1].scatter([1,2,3], [1,4,9], color='black')
axes[1].set_title('Slope changes by location')
for ax in axes:
    ax.axhline(0, color='gray', lw=.6)
    ax.axvline(0, color='gray', lw=.6)
    ax.legend(); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()

slope1 = (4-1)/(2-1)
slope2 = (9-4)/(3-2)
print('첫 구간 평균 기울기:', slope1)   # 3.0
print('둘째 구간 평균 기울기:', slope2) # 5.0
```

## 4. AI와 연결: 손실이라는 함수

어떤 모델의 가중치를 `w`라 하고, 예측이 틀린 정도를 `L(w)=(w-3)²`로 정의해 보자. `w=3`에서 손실이 최소가 된다. 학습 알고리즘은 **손실 그래프에서 내려가는 방향**을 찾아 가중치를 조정한다. 그래서 함수의 기울기를 알아야 한다.

> 흔한 오해: 기울기는 언제나 출력값과 같지 않다. `f(2)=4`와 `x=2`에서의 기울기는 우연히 같을 수도, 다를 수도 있다.

---

## 직접 풀어보기

① `f(x)=3x-2`에서 `f(4)`와 기울기를 계산하라. ② `g(x)=x²`의 `[3,4]`와 `[4,5]` 평균 기울기를 구하라. ③ `L(w)=(w-3)²`가 최소가 되는 가중치는?

<details>
<summary>정답·힌트 확인</summary>

① `10`, 기울기 `3`. ② `(16-9)/(4-3)=7`, `(25-16)/(5-4)=9`. ③ `w=3`; 손실 0.

</details>

## 미션 평가 체크

- [ ] 손실 함수를 그래프로 해석하고 이후 경사하강법의 가중치 이동 방향을 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 1-5 특이값 분해(SVD)](../STEP_1_선형대수/STEP_1-5_특이값_분해_SVD.md) · [전체 목차](../README.md) · [다음: 2-2 극한과 미분 →](STEP_2-2_극한과_미분.md)
