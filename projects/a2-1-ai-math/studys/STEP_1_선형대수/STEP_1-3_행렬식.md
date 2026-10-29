# STEP 1-3. 행렬식(Determinant): 변환으로 면적이 얼마나 달라질까?

> A2-1 AI 수학 · STEP 1 선형대수 · 난이도: 초급 · 권장 학습 시간: 90~120분

## 0. 이번 학습 목표

- 2×2 행렬식 `ad - bc`를 손으로 계산한다.
- `abs(det(A))`가 **2차원 도형의 면적 배율**이라는 뜻을 설명한다.
- `det(A)`의 부호와 0의 의미를 이해한다.
- NumPy 및 **신발끈 공식(Shoelace formula)**으로 면적비를 검증한다.
- A2-1 평가 기준인 **면적비 오차 1% 이내**를 확인한다.

```bash
python -m pip install numpy matplotlib
```

---

## 1. 직관: 고무판 위의 정사각형

가로 1, 세로 1인 정사각형을 고무판에 그렸다고 상상해 보자. 고무판을 가로 2배, 세로 3배 늘리면 면적은 원래 1에서 6으로 변한다.

\[
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\]

행렬식은 이 면적 변화의 배율을 나타낸다.

\[
\det(A)=2\times3=6
\]

> **핵심:** 2차원에서 `abs(det(A)) = 변환된 면적 / 원래 면적`이다. 단, 원래 도형의 면적이 0이 아니어야 한다.

## 2. 2×2 행렬식 계산법

\[
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
\quad\Rightarrow\quad
\det(A)=ad-bc
\]

예를 들어

\[
A=\begin{bmatrix}2&1\\1&3\end{bmatrix}
\]

이면

\[
\det(A)=2\times3-1\times1=5
\]

이다. 그러므로 **원래 면적이 1인 도형**은 이 행렬로 변환한 뒤 면적이 5가 된다.

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
manual = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
result = np.linalg.det(A)
print("손계산:", manual)  # 5.0
print("NumPy:", result)  # 5.000... (부동소수점 오차 가능)
print("일치:", np.isclose(manual, result))  # True
```

## 3. 행렬식의 부호가 말하는 것

| 행렬식 | 의미 | 쉬운 예 |
|---|---|---|
| `det(A) > 0` | 방향(orientation)을 보존 | 회전, 양의 축 스케일링 |
| `det(A) < 0` | 방향이 뒤집힘 | x축 또는 y축 방향 반사 |
| `det(A) = 0` | 면적이 0으로 찌그러짐 | 평면을 선이나 점으로 압축 |

**회전**과 **전단**의 행렬식은 1이다(2차원 표준 회전·전단 기준). 따라서 도형의 생김새나 방향은 변하더라도 면적은 보존한다.

반면 다음 행렬은 y값을 모두 0으로 만든다.

\[
A_{flat}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad \det(A_{flat})=0
\]

면적이 0으로 되므로 역변환을 할 수 없다. 이 때문에 **행렬식이 0인 정사각행렬은 역행렬이 없다.**

## 4. 직접 검증: 신발끈 공식

행렬식은 **예상 면적비**를 알려준다. 실제 도형 넓이도 따로 계산해서 비교해 보자.

신발끈 공식은 순서대로 나열한 다각형 꼭짓점 `(x_i, y_i)`를 이용한다.

\[
\mathrm{Area}=\frac{1}{2}\left|\sum_{i=0}^{n-1}
(x_i y_{i+1}-y_i x_{i+1})\right|
\]

마지막 꼭짓점 다음은 첫 꼭짓점으로 돌아온다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 꼭짓점을 반시계 방향으로 나열한다. 마지막 꼭짓점 중복 불필요.
square = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])  # shape: (4, 2), 각 행이 한 꼭짓점

A = np.array([[2.0, 1.0], [1.0, 3.0]])
transformed = square @ A.T  # (4, 2) @ (2, 2) -> (4, 2)

def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1)))

original_area = polygon_area(square)
new_area = polygon_area(transformed)
actual_ratio = new_area / original_area
expected_ratio = abs(np.linalg.det(A))
error_percent = abs(actual_ratio - expected_ratio) / expected_ratio * 100

print(f"원본 면적: {original_area:.6f}")
print(f"변환 면적: {new_area:.6f}")
print(f"실제 면적비: {actual_ratio:.6f}")
print(f"|det(A)|: {expected_ratio:.6f}")
print(f"상대 오차: {error_percent:.8f}%")
print("1% 이내:", error_percent <= 1.0)

# 원본과 변환 결과를 한 그래프에 표시
fig, ax = plt.subplots(figsize=(7, 6))
for pts, label in [(square, "Original"), (transformed, "Transformed")]:
    closed = np.vstack([pts, pts[0]])
    ax.fill(closed[:, 0], closed[:, 1], alpha=0.18)
    ax.plot(closed[:, 0], closed[:, 1], "o-", label=label)

ax.axhline(0, color="gray", linewidth=0.6)
ax.axvline(0, color="gray", linewidth=0.6)
ax.set_aspect("equal")
ax.set_title(f"Area ratio={actual_ratio:.2f}, |det(A)|={expected_ratio:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()
```

**예상 출력** (부동소수점 반올림에 따라 마지막 자릿수는 달라질 수 있음):

```text
원본 면적: 1.000000
변환 면적: 5.000000
실제 면적비: 5.000000
|det(A)|: 5.000000
상대 오차: 0.00000000%
1% 이내: True
```

### 왜 `square @ A.T`일까?

STEP 1-2에서는 점을 **열벡터**로 모아 `A @ points`로 계산했다. 여기서는 점을 **행벡터**로 저장했으므로 `points @ A.T`로 계산해야 같은 변환이 된다.

- 열벡터 모음: `(2, N)` → `A @ points`
- 행벡터 모음: `(N, 2)` → `points @ A.T`

이 차이는 NumPy 사용 시 특히 중요하다.

## 5. 특수한 세 가지 행렬의 면적비

```python
import numpy as np

theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2.0, 0.0], [0.0, 0.5]])
H = np.array([[1.0, 1.0], [0.0, 1.0]])

for name, matrix in [("회전", R), ("스케일링", S), ("전단", H)]:
    print(name, "det =", round(float(np.linalg.det(matrix)), 6))

# 예상: 회전 1.0, 스케일링 1.0, 전단 1.0
```

여기서 스케일링이 흥미롭다. 가로를 2배로 키우고 세로를 절반으로 줄여 `2 × 0.5 = 1`이기 때문에 **도형 모양은 바뀌어도 면적은 동일**하다.

## 6. 수치 검증할 때 주의할 점

- **절댓값 사용:** 면적비는 음수가 될 수 없으므로 `abs(det(A))`와 비교한다.
- **상대 오차의 분모:** `expected_ratio`가 0이라면 0으로 나누게 된다. 이 경우는 별도로 절대 오차를 비교해야 한다.
- **부동소수점:** `5.000000000000001`처럼 미세한 차이가 정상이다. `==` 대신 `np.isclose`를 쓴다.
- **꼭짓점 순서:** 신발끈 공식은 다각형 경계를 따라 순서대로 꼭짓점을 넣어야 한다.

면적이 0으로 붕괴하는 경우를 별도로 테스트하려면:

```python
flat = np.array([[1.0, 0.0], [0.0, 0.0]])
print(np.isclose(np.linalg.det(flat), 0.0))  # True
```

## 7. A2-1 평가 연결

평가 문항: **`det(A)` 기반 면적비 검증 결과의 오차가 1% 이내로 출력되어 있는가?**

검증할 내용은 세 가지다.

1. `np.linalg.det(A)`로 **이론 면적 배율**을 구했다.
2. 신발끈 공식으로 **실제 변환 전후 면적비**를 구했다.
3. `상대 오차(%)`와 `1% 이내` 여부를 **콘솔에 출력**했다.

> 면적비의 이론값과 실제값을 서로 다른 방식으로 계산해야 의미 있는 검증이다. `변환 면적 = 원본 면적 * abs(det(A))`로 두 값을 만들면 독립 검증이 아니다.

## 8. 연습문제

**문제 1.** `A = [[3, 0], [0, 2]]`의 행렬식과 면적비는?

**문제 2.** `A = [[-1, 0], [0, 1]]`의 행렬식은 음수다. 면적은 음수가 되는가?

**문제 3.** `A = [[2, 4], [1, 2]]`는 왜 역행렬이 없을까?

**문제 4.** 4장 코드의 `A`를 전단 행렬 `[[1, 2], [0, 1]]`로 바꾸고 오차를 출력해 보자.

<details>
<summary>정답 확인</summary>

1. `det(A)=6`, 면적비 `6`
2. 아니다. 행렬식은 `-1`이고 면적비는 `|-1|=1`이다.
3. `det(A)=2×2−4×1=0`이므로 면적이 0으로 붕괴한다.
4. `det(A)=1`, 이론·실제 면적비가 약 1, 오차 약 0%

</details>

## 9. 완료 체크리스트

- [ ] 2×2 행렬식을 손으로 계산한다.
- [ ] 행렬식의 양수·음수·0을 설명한다.
- [ ] `abs(det(A))`와 실제 면적비를 구분한다.
- [ ] 신발끈 공식으로 면적 검증을 실행한다.
- [ ] 상대 오차 1% 이내를 출력하고 의미를 설명한다.

**다음 학습:** [STEP 1-4. 고유값과 고유벡터](STEP_1-4_고유값과_고유벡터.md) — 변환해도 방향이 유지되는 특별한 벡터를 찾는다.
