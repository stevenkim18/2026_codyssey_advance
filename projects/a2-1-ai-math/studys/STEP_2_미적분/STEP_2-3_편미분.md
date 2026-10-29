# STEP 2-3. 편미분

> 코디세이 A2-1 AI 수학 · STEP 2 미적분 · Python/NumPy 입문자용

## 학습 목표

- 입력 변수가 여럿인 함수를 이해한다.
- 다른 변수를 고정하고 한 변수만 미분한다.
- 가중치별 기울기를 계산하는 이유를 이해한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 직관: 손실은 하나의 숫자지만 가중치는 많다

모델의 결과는 가중치 `w₁`, `w₂`, 편향 `b` 등에 따라 바뀐다. 손실도 이들의 함수이므로 `L(w₁,w₂)`처럼 **입력이 두 개인 함수**가 된다. `w₁`만 조금 바꾼 영향, `w₂`만 조금 바꾼 영향을 각각 알고 싶다.

## 2. 편미분: 나머지는 잠시 고정

\[f(x,y)=x^2+3xy+2y^2\]

`x`로 편미분하면 `y`는 숫자 상수로 취급한다.

\[\frac{\partial f}{\partial x}=2x+3y\]

`y`로 편미분하면 `x`를 상수로 취급한다.

\[\frac{\partial f}{\partial y}=3x+4y\]

점 `(x,y)=(2,1)`에서는 각각 `7`과 `10`이다. `∂` 기호는 **편미분**을 의미하며 일반적인 `d`와 달리 특정 입력 방향을 강조한다.

## 3. NumPy 실습: 직접 변화시켜 검증하기

```python
import numpy as np

def f(x, y): return x*x + 3*x*y + 2*y*y

def partial_x(x, y, h=1e-5):
    return (f(x+h, y)-f(x-h, y))/(2*h)

def partial_y(x, y, h=1e-5):
    return (f(x, y+h)-f(x, y-h))/(2*h)

x, y = 2.0, 1.0
analytic = np.array([2*x+3*y, 3*x+4*y])
numeric = np.array([partial_x(x,y), partial_y(x,y)])
print('손계산:', analytic)  # [7. 10.]
print('수치미분:', numeric)
print('최대 오차:', np.max(np.abs(analytic-numeric)))
assert np.allclose(analytic, numeric, atol=1e-4)
```

## 4. 머신러닝 사례

예측 모델 `ŷ=w₁x₁+w₂x₂+b`에서 데이터가 한 개라면 MSE형 손실은 `L=(ŷ-y)²`이다. 연쇄 법칙을 사용하면

\[\frac{\partial L}{\partial w_1}=2(\hat y-y)x_1,\quad
\frac{\partial L}{\partial w_2}=2(\hat y-y)x_2,\quad
\frac{\partial L}{\partial b}=2(\hat y-y)\]

가 된다. 결국 **각 매개변수가 손실에 얼마만큼 책임이 있는지** 계산한 것이다.

> 혼동 주의: 편미분을 하면 나머지 변수 값이 0으로 사라지는 것이 아니라, 미분 시 상수로 취급된다.

---

## 직접 풀어보기

① `g(x,y)=2x²+xy+4y`의 `∂g/∂x`, `∂g/∂y`를 구하라. ② `(1,2)`에서 각 값을 계산하라. ③ 위 코드에서 함수를 바꿔 수치값과 비교하라.

<details>
<summary>정답·힌트 확인</summary>

① `4x+y`, `x+4`. ② `6`, `5`.

</details>

## 미션 평가 체크

- [ ] 모델의 여러 가중치에 편미분을 적용하는 이유를 직접 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 2-2 극한과 미분](STEP_2-2_극한과_미분.md) · [전체 목차](../README.md) · [다음: 2-4 Gradient(기울기 벡터) →](STEP_2-4_Gradient기울기_벡터.md)
