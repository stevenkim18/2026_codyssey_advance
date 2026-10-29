# STEP 2-2. 극한과 미분

> 코디세이 A2-1 AI 수학 · STEP 2 미적분 · Python/NumPy 입문자용

## 학습 목표

- 평균 변화율에서 순간 변화율로 넘어가는 이유를 이해한다.
- 도함수를 손으로 구하고 중심차분으로 확인한다.
- 미분 오차와 h의 관계를 실험한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 직관: 자동차 속도계는 무엇을 보여 줄까?

10초 동안 100m 이동하면 평균 속도는 10m/s다. 그러나 **지금 이 순간**의 속도를 알고 싶으면 관측 시간을 아주 짧게 줄여야 한다. 그래프의 평균 기울기를 한 점에서의 접선 기울기로 바꾸는 아이디어가 미분이다.

\[f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}\]

`h`는 작은 이동 거리이고 `lim`은 `h`가 0에 가까워질 때의 값을 뜻한다. 분모가 실제로 0이라는 뜻은 아니다.

## 2. 손계산: f(x)=x²

\[\frac{(x+h)^2-x^2}{h}=\frac{2xh+h^2}{h}=2x+h\]

`h`를 0에 가깝게 하면 `f'(x)=2x`. 따라서 `x=3`에서 접선 기울기는 `6`이다. 연습으로 `d(x³)/dx=3x²`도 기억해 두자.

## 3. 컴퓨터에서의 미분: 전진차분과 중심차분

전진차분 `(f(x+h)-f(x))/h`는 한쪽 정보만 쓰지만, **중심차분** `(f(x+h)-f(x-h))/(2h)`는 양쪽 정보를 사용한다. 매끄러운 함수에서는 일반적으로 중심차분이 더 정확하다. 다만 `h`가 지나치게 작으면 부동소수점 반올림이 문제가 된다.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**3

def forward(f, x, h): return (f(x+h)-f(x))/h

def central(f, x, h): return (f(x+h)-f(x-h))/(2*h)

x0 = 3.0
exact = 3*x0**2  # 27
hs = np.logspace(-1, -8, 30)
err_forward = [abs(forward(f, x0, h)-exact) for h in hs]
err_central = [abs(central(f, x0, h)-exact) for h in hs]
print('정확한 도함수:', exact)
print('중심차분:', central(f, x0, 1e-5))
print('오차:', abs(central(f, x0, 1e-5)-exact))
assert abs(central(f, x0, 1e-5)-exact) < 1e-4

plt.loglog(hs, err_forward, 'o-', label='Forward')
plt.loglog(hs, err_central, 'o-', label='Central')
plt.xlabel('h'); plt.ylabel('Absolute error')
plt.title('Numerical differentiation error'); plt.legend()
plt.grid(True, which='both', alpha=.2); plt.show()
```

## 4. 왜 AI에 필요한가?

모델의 손실이 `L(w)`라면 `dL/dw`는 **가중치를 조금 바꿨을 때 손실이 어떻게 반응하는지** 알려 준다. 예를 들어 `L(w)=(w-3)²`이면 `dL/dw=2(w-3)`이고 `w=1`에서 `-4`다. 손실을 줄이려면 `w`를 **증가시키는 쪽**으로 움직여야 한다.

> 중요한 구분: 수치미분은 작은 차이를 이용해 기울기를 근사하는 방법이다. 역전파는 연쇄 법칙을 활용해 미분을 효율적으로 계산한다.

---

## 직접 풀어보기

① `f(x)=x⁴`의 도함수를 구하라. ② `x=2`에서 그 기울기는? ③ `h=1e-3`과 `1e-5`의 중심차분 오차를 비교하라.

<details>
<summary>정답·힌트 확인</summary>

① `f'(x)=4x³`. ② `32`. ③ 해당 함수에서는 일반적으로 `1e-5`가 더 정확하지만, 무한히 작게 한다고 항상 좋아지는 것은 아니다.

</details>

## 미션 평가 체크

- [ ] 수치미분과 해석적 도함수 차이가 `1e-4` 이하인지 출력하고 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 2-1 함수와 기울기](STEP_2-1_함수와_기울기.md) · [전체 목차](../README.md) · [다음: 2-3 편미분 →](STEP_2-3_편미분.md)
