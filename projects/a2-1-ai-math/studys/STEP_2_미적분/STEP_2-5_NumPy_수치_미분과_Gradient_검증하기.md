# STEP 2-5. NumPy 수치 미분과 Gradient 검증하기

> 코디세이 A2-1 AI 수학 · STEP 2 미적분 · Python/NumPy 입문자용

## 학습 목표

- 중심차분과 해석적 미분을 비교한다.
- 수치 Gradient를 구현하고 오차를 평가한다.
- A2-1 미적분 평가 기준을 재현한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 왜 검증해야 할까?

손으로 구한 도함수에는 부호나 계수를 잘못 쓸 수 있다. NumPy로 작은 변화를 실제 적용해 얻은 **수치미분**과 비교하면 실수를 발견하기 쉽다. 역전파에서도 똑같은 원리의 **Gradient Checking**을 사용한다.

## 2. 중심차분 공식

\[f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}\]

\[\frac{\partial f}{\partial x_i}\approx\frac{f(\mathbf x+h\mathbf e_i)-f(\mathbf x-h\mathbf e_i)}{2h}\]

`e_i`는 `i`번째 성분만 1인 단위벡터다. 즉, **한 번에 한 좌표만 바꿔** 손실 변화량을 측정한다.

## 3. 미션 제출용 재현 코드

```python
import numpy as np
import matplotlib.pyplot as plt

h = 1e-5
f = lambda x: x**3
x = 3.0
numeric = (f(x+h)-f(x-h))/(2*h)
analytic = 3*x*x
error = abs(numeric-analytic)
print(f'[미분] 수치={numeric:.8f}, 해석={analytic:.8f}, 오차={error:.2e}')
assert error < 1e-4

# 일반 N차원 Gradient: 함수는 벡터를 받아 스칼라 반환

def numerical_gradient(func, vector, h=1e-5):
    vector = np.asarray(vector, dtype=float)
    result = np.zeros_like(vector)
    for i in range(vector.size):
        plus, minus = vector.copy(), vector.copy()
        plus.flat[i] += h; minus.flat[i] -= h
        result.flat[i] = (func(plus)-func(minus))/(2*h)
    return result

loss = lambda p: (p[0]-1)**2 + 2*(p[1]+2)**2
point = np.array([2.0, 0.0])
exact_grad = np.array([2*(point[0]-1), 4*(point[1]+2)])
computed_grad = numerical_gradient(loss, point)
print('해석 Gradient:', exact_grad)
print('수치 Gradient:', computed_grad)
print('Gradient 최대 오차:', np.max(np.abs(exact_grad-computed_grad)))
assert np.allclose(exact_grad, computed_grad, atol=1e-4)

xx, yy = np.meshgrid(np.linspace(-2,4,150), np.linspace(-5,1,150))
plt.contour(xx,yy,(xx-1)**2+2*(yy+2)**2,levels=14)
px, py = np.meshgrid(np.linspace(-1.5,3.5,7),np.linspace(-4.5,.5,7))
g1, g2 = 2*(px-1), 4*(py+2)
length = np.hypot(g1,g2)
plt.quiver(px,py, np.divide(g1,length,out=np.zeros_like(g1),where=length>0),
           np.divide(g2,length,out=np.zeros_like(g2),where=length>0), color='crimson')
plt.axis('equal'); plt.title('Gradient and contour'); plt.show()
```

## 4. 결과 읽는 법

- **해석값**은 미분 공식으로 구한 값이다.
- **수치값**은 작은 `h`로 차이를 측정한 값이다.
- **절대 오차**가 `1e-4`보다 작아야 미션 기준을 만족한다.
- `plt.quiver()`는 화살표를, `plt.contour()`는 등고선을 그린다.

> 수치미분은 검증 도구로 훌륭하지만 모델의 파라미터가 수백만 개라면 매우 느리다. 그때 역전파가 필요하다.

---

## 직접 풀어보기

① `f(x)=sin(x)`의 도함수를 `cos(x)`와 비교하라. ② 위 Gradient 함수를 `p`가 세 성분인 함수에 적용해 보라. ③ `h`를 `1e-2`, `1e-5`, `1e-10`으로 바꾸며 오차를 기록하라.

<details>
<summary>정답·힌트 확인</summary>

① `np.sin`, `np.cos`를 이용해 `x=0.7`에서 비교하면 된다. ② `loss=lambda p: np.sum(p**2)`이면 Gradient는 `2*p`. ③ 너무 큰 `h`와 너무 작은 `h` 모두 오차가 커질 수 있다.

</details>

## 미션 평가 체크

- [ ] **오차 <1e-4 출력**, 등고선에 수직인 Gradient 화살표 출력, 두 가지를 모두 완료했는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 2-4 Gradient(기울기 벡터)](STEP_2-4_Gradient기울기_벡터.md) · [전체 목차](../README.md) · [다음: 3-1 퍼셉트론과 인공신경망 →](../STEP_3_신경망_역전파/STEP_3-1_퍼셉트론과_인공신경망.md)
