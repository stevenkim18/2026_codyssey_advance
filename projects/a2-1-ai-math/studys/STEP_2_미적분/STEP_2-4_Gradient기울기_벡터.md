# STEP 2-4. Gradient(기울기 벡터)

> 코디세이 A2-1 AI 수학 · STEP 2 미적분 · Python/NumPy 입문자용

## 학습 목표

- 여러 편미분을 하나의 벡터로 묶는다.
- Gradient가 손실의 가장 가파른 증가 방향인 이유를 이해한다.
- 등고선에 수직으로 놓인 Gradient 화살표를 시각화한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 등산 지도 비유

등고선은 높이가 같은 점들을 연결한 선이다. 같은 등고선을 따라 움직이면 높이는 변하지 않는다. 반대로 가장 빠르게 고도를 높이는 방향으로 움직이면 등고선에 수직으로 건너가게 된다. 이것이 Gradient(그라디언트)다.

## 2. 정의

\[\nabla f(x,y)=\begin{bmatrix}\partial f/\partial x\\\partial f/\partial y\end{bmatrix}\]

`f(x,y)=(x-1)²+2(y+2)²`라면

\[\nabla f(x,y)=\begin{bmatrix}2(x-1)\\4(y+2)\end{bmatrix}\]

점 `(2,0)`에서 Gradient는 `(2,8)`이며, 가장 빠르게 증가하는 방향을 가리킨다. **손실을 줄이는 방향**은 반대 방향인 `-∇f`다.

## 3. 등고선과 화살표 함께 그리기

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x,y): return (x-1)**2 + 2*(y+2)**2

def grad(x,y): return 2*(x-1), 4*(y+2)

xx, yy = np.meshgrid(np.linspace(-3, 5, 250), np.linspace(-6, 2, 250))
z = f(xx, yy)
qx, qy = np.meshgrid(np.linspace(-2.5, 4.5, 9), np.linspace(-5.5, 1.5, 9))
g_x, g_y = grad(qx, qy)
length = np.hypot(g_x, g_y)
# 방향 비교가 쉽도록 0이 아닌 벡터만 길이를 1로 맞춘다.
u = np.divide(g_x, length, out=np.zeros_like(g_x), where=length>0)
v = np.divide(g_y, length, out=np.zeros_like(g_y), where=length>0)
plt.figure(figsize=(7,6))
plt.contour(xx, yy, z, levels=15)
plt.quiver(qx, qy, u, v, color='red', pivot='mid')
plt.scatter([1],[-2],color='black', label='minimum (1,-2)')
plt.axis('equal'); plt.title('Gradient is normal to contour lines')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
print('grad(2,0)=', grad(2,0))   # (2, 8)
```

**왜 수직일까?** 등고선을 따라 움직이는 미소 변위 `d=(dx,dy)`에서는 함수값 변화가 0이므로 `∇f · d = 0`이다. 내적이 0이니 서로 수직이다. 단, Gradient가 0인 최솟값에서는 화살표에 방향이 없다.

## 4. AI 학습식과 연결

\[\mathbf w_{t+1}=\mathbf w_t-\alpha\nabla L(\mathbf w_t)\]

`α`는 학습률이다. 음의 Gradient 쪽으로 이동하면 **충분히 작은 한 걸음에서는** 손실이 감소할 가능성이 높다. 너무 큰 걸음은 오히려 손실을 증가시킬 수도 있다.

---

## 직접 풀어보기

① `f(x,y)=x²+y²`의 Gradient는? ② `(3,4)`에서 가장 빨리 감소하는 방향은? ③ 등고선 접선 방향 `(-4,3)`과 Gradient의 내적을 계산하라.

<details>
<summary>정답·힌트 확인</summary>

① `(2x,2y)` ② 음의 Gradient `(-6,-8)` 방향 ③ `(6,8)·(-4,3)=-24+24=0`, 서로 수직.

</details>

## 미션 평가 체크

- [ ] Gradient 화살표가 **등고선에 수직**으로 표시되며, 최솟값에서 크기가 0이라는 점을 설명할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 2-3 편미분](STEP_2-3_편미분.md) · [전체 목차](../README.md) · [다음: 2-5 NumPy 수치 미분과 Gradient 검증하기 →](STEP_2-5_NumPy_수치_미분과_Gradient_검증하기.md)
