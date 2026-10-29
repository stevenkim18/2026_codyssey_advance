# STEP 1-2. 선형 변환: 회전·스케일링·전단을 눈으로 이해하기

> A2-1 AI 수학 · STEP 1 선형대수 · 난이도: 입문~초급 · 권장 학습 시간: 90~120분

## 0. 이번 학습 목표

- `A @ v`가 좌표를 변환한다는 사실을 설명한다.
- **회전(Rotation), 스케일링(Scaling), 전단(Shear)** 행렬을 구성한다.
- NumPy와 Matplotlib으로 **변환 그래프 3종**을 한 Figure에 출력한다.
- 행렬 곱셈의 **순서가 결과에 영향**을 준다는 것을 확인한다.

```bash
python -m pip install numpy matplotlib
```

---

## 1. 직관: 투명 필름 위의 그림을 움직이기

종이에 원을 그린 뒤 투명 필름 위에 복사한다고 상상해 보자. 필름을 돌리면 **회전**, 가로로 늘리면 **스케일링**, 윗부분을 옆으로 밀면 **전단**이다.

좌표 하나하나를 계산할 수도 있지만, **행렬 하나**로 모든 점에 같은 규칙을 적용할 수 있다.

\[
\begin{bmatrix}x'\\y'\end{bmatrix}
=A\begin{bmatrix}x\\y\end{bmatrix}
\]

원 위 점이 100개라면 점 100개에 똑같은 `A`를 적용한다.

## 2. 스케일링: 늘리거나 줄이기

\[
S=\begin{bmatrix}s_x&0\\0&s_y\end{bmatrix}
\]

`S = [[2, 0], [0, 0.5]]`를 적용하면 x는 2배, y는 0.5배가 된다.

\[
S\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}2\\1\end{bmatrix}
\]

```python
import numpy as np
S = np.array([[2.0, 0.0], [0.0, 0.5]])
print(S @ np.array([1.0, 2.0]))  # [2. 1.]
```

> 두 축을 같은 비율로 키우면 모양은 유지되고 크기만 바뀐다. 비율이 다르면 원이 타원으로 보인다.

## 3. 회전: 방향만 바꾸기

원점을 중심으로 반시계 방향으로 \(\theta\)만큼 회전하는 행렬은

\[
R(\theta)=\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\]

이다. 예를 들어 90° 회전하면 `(1, 0)`은 `(0, 1)`이 된다.

```python
angle = np.deg2rad(90)   # np.sin, np.cos는 라디안을 사용
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
print(np.round(R @ np.array([1.0, 0.0]), 6))  # [0. 1.]
```

`np.deg2rad(45)`를 사용하면 45°를 라디안으로 바꿀 수 있다.

## 4. 전단: 평행사변형처럼 기울이기

x방향 전단 행렬은 다음과 같다.

\[
H=\begin{bmatrix}1&k\\0&1\end{bmatrix}
\quad\Rightarrow\quad x'=x+ky,\quad y'=y
\]

`k=1`이면 원래 y가 2였던 점은 x좌표가 2만큼 더해진다.

```python
H = np.array([[1.0, 1.0], [0.0, 1.0]])
print(H @ np.array([1.0, 2.0]))  # [3. 2.]
```

전단은 단순한 회전이 아니다. 높이가 클수록 가로로 더 많이 밀려 모양이 비스듬해진다.

## 5. 핵심 실습: 변환 그래프 3종 만들기

아래 코드는 **완전히 독립적으로 실행**된다. 원과 정사각형을 함께 그려 원점 기준 변환을 확인한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# (1) 변환할 데이터 준비: 2×N 형식 (열 하나가 점 하나)
t = np.linspace(0, 2 * np.pi, 300)
circle = np.vstack([np.cos(t), np.sin(t)])

square = np.array([
    [-1, 1, 1, -1, -1],  # x좌표
    [-1, -1, 1, 1, -1]   # y좌표
], dtype=float)

# (2) 변환 행렬
angle = np.deg2rad(45)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
S = np.array([[2.0, 0.0], [0.0, 0.5]])
H = np.array([[1.0, 1.0], [0.0, 1.0]])

# (3) A @ points: 모든 열벡터에 행렬 A 적용
transforms = [
    ("Rotation (45 deg)", R),
    ("Scaling (x2, y0.5)", S),
    ("Shear (k=1)", H)
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, A) in zip(axes, transforms):
    ax.plot(circle[0], circle[1], "--", color="gray", label="Original circle")
    ax.plot(square[0], square[1], ":", color="gray", label="Original square")

    new_circle = A @ circle
    new_square = A @ square
    ax.plot(new_circle[0], new_circle[1], label="Transformed circle")
    ax.plot(new_square[0], new_square[1], label="Transformed square")

    ax.axhline(0, color="lightgray")
    ax.axvline(0, color="lightgray")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title(name)
    ax.legend(fontsize=8, loc="lower left")

plt.suptitle("Three Linear Transformations")
plt.tight_layout()
plt.show()
```

### 출력에서 확인할 것

| 변환 | 원래 원 | 원래 정사각형 | 중요한 변화 |
|---|---|---|---|
| 회전 | 크기가 같은 원 | 기울어진 정사각형 | 방향만 변경 |
| 스케일링 | 가로로 늘어난 타원 | 가로 2배·세로 절반 | x/y 길이 변화 |
| 전단 | 기울어진 타원 | 평행사변형 | 높이에 따라 x가 이동 |

## 6. 행렬 곱셈 순서는 왜 중요할까?

`R @ S @ v`와 `S @ R @ v`는 보통 다르다.

- `R @ (S @ v)`는 **스케일링 후 회전**
- `S @ (R @ v)`는 **회전 후 스케일링**

```python
v = np.array([1.0, 1.0])
print("scale → rotate:", np.round(R @ S @ v, 3))
print("rotate → scale:", np.round(S @ R @ v, 3))
print("같은가?", np.allclose(R @ S @ v, S @ R @ v))  # False
```

파이썬의 `@`는 일반적인 행렬 곱과 같은 순서를 따른다. **오른쪽 연산이 벡터에 먼저 적용**된다.

## 7. '선형'이라는 말의 뜻

변환 `T`가 선형이면 다음 두 조건을 만족한다.

\[
T(\mathbf{u}+\mathbf{v})=T(\mathbf{u})+T(\mathbf{v}),\qquad
T(c\mathbf{u})=cT(\mathbf{u})
\]

쉽게 말해 **더한 뒤 변환**하거나 **변환한 뒤 더해도** 같은 결과다. 또한 선형 변환은 항상 원점 `(0,0)`을 원점에 둔다. 평행이동 `T(v)=v+b`(b가 0이 아닐 때)는 엄밀하게는 선형이 아니라 *아핀 변환*이다.

```python
u = np.array([1.0, 2.0])
v = np.array([-1.0, 3.0])
print(np.allclose(H @ (u + v), H @ u + H @ v))  # True
```

## 8. A2-1 미션 평가 연결

미션의 선형대수 항목 중 **회전/스케일링/전단 변환 그래프 3종**이 이 단원에 해당한다.

- [ ] Python 실행 시 변환 3종 그래프가 모두 표시된다.
- [ ] 원본과 변환 결과가 구별된다.
- [ ] 회전 각도, 스케일 비율, 전단 계수를 설명할 수 있다.
- [ ] 행렬 곱셈 순서가 중요한 이유를 예로 설명할 수 있다.

## 9. 연습문제

**문제 1.** 원 위의 점 `(1, 0)`을 180° 회전하면 어디로 갈까?

**문제 2.** `S = [[3, 0], [0, 2]]`를 점 `(2, 4)`에 적용하면?

**문제 3.** `H = [[1, -0.5], [0, 1]]`이면 `(2, 4)`는 어디로 갈까?

**문제 4.** 실습 코드에서 `angle=30`, `k=-0.7`로 바꿔 그래프를 다시 그려 보자.

<details>
<summary>정답 확인</summary>

1. `(-1, 0)` (부동소수점으로 매우 작은 오차 가능)
2. `(6, 8)`
3. `(0, 4)`
4. 직접 그래프로 확인. 전단 방향이 반대로 기울어진다.

</details>

**다음 학습:** [STEP 1-3. 행렬식](STEP_1-3_행렬식.md) — 이 변환이 도형의 넓이를 몇 배 바꾸는지 계산한다.
