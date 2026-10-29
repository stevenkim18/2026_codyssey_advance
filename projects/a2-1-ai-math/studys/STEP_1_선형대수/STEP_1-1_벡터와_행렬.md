# STEP 1-1. 벡터와 행렬: AI가 숫자를 다루는 기본 언어

> A2-1 AI 수학 · STEP 1 선형대수 · 난이도: 입문 · 권장 학습 시간: 60~90분

## 0. 이 수업에서 배울 것

- 벡터를 **숫자로 나타낸 위치·방향·특징의 묶음**으로 이해한다.
- 행렬을 **숫자 표이면서 벡터를 변환하는 규칙**으로 이해한다.
- `shape`, `@`, `*`, `np.linalg.norm()`의 역할을 설명한다.
- 행렬 곱셈을 손으로 계산한 뒤 NumPy 결과와 비교한다.

### 준비

```bash
python -m pip install numpy matplotlib
```

코드를 `.py`에 복사해 실행하거나 Jupyter Notebook의 셀에 입력해도 된다.

---

## 1. 벡터란?

게임 캐릭터가 오른쪽으로 3칸, 위로 2칸 이동한다고 생각해 보자.

\[
\mathbf{v}=\begin{bmatrix}3\\2\end{bmatrix}
\]

이 벡터에는 **두 숫자가 한 묶음**으로 들어 있다. 그림에서는 원점 `(0, 0)`에서 `(3, 2)`로 향하는 화살표로 볼 수 있다.

다른 관점으로는 한 사람의 정보 `[키, 몸무게]`, 상품의 특징 `[가격, 평점, 재고]`도 벡터다. AI에서는 입력값이나 가중치를 벡터로 표현한다.

```python
import numpy as np

v = np.array([3, 2])
w = np.array([1, -1])

print("v:", v)               # [3 2]
print("v의 크기(shape):", v.shape)  # (2,)
print("벡터 덧셈:", v + w)    # [4 1]
print("2배:", 2 * v)        # [6 4]
print("길이:", np.linalg.norm(v))  # sqrt(13) ≈ 3.606
```

> `shape == (2,)`는 숫자 2개로 이루어진 **1차원 배열**이다. `(2, 1)`인 열벡터 형태와 NumPy에서는 구분한다.

### 벡터 길이(노름)

\[
\|\mathbf{v}\|=\sqrt{3^2+2^2}=\sqrt{13}\approx3.606
\]

피타고라스 정리로 화살표 길이를 계산하는 것이다. 이후 **고유벡터를 길이 1로 정규화**할 때 쓴다.

## 2. 행렬이란?

행렬은 숫자를 행과 열로 배열한 것이다.

\[
A=\begin{bmatrix}1&2\\3&4\end{bmatrix}
\]

- `1, 2`는 첫 번째 **행(row)**
- `1, 3`은 첫 번째 **열(column)**
- 이 행렬의 크기는 **2행 × 2열**, NumPy에서는 `(2, 2)`

```python
A = np.array([
    [1, 2],
    [3, 4]
])

print(A.shape)       # (2, 2)
print(A[0, 1])       # 2: 0번째 행, 1번째 열
print(A[:, 0])       # [1 3]: 첫 번째 열
print(A.T)           # 전치행렬: 행과 열을 맞바꿈
```

## 3. 행렬 × 벡터: 가장 중요한 계산

\[
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad
\mathbf{x}=\begin{bmatrix}2\\1\end{bmatrix}
\]

\[
A\mathbf{x}=
\begin{bmatrix}1\times2+2\times1\\3\times2+4\times1\end{bmatrix}
=\begin{bmatrix}4\\10\end{bmatrix}
\]

첫 번째 결과 `4`는 첫 번째 행 `[1, 2]`와 입력 `[2, 1]`을 각각 곱해 더한 값이다. 두 번째 결과 `10`도 같은 방법이다.

```python
x = np.array([2, 1])
y = A @ x

print(y)                     # [ 4 10]
print(np.allclose(y, [4, 10]))  # True
```

`@`는 **행렬 곱셈**이다. `*`는 원소별 곱셈이라 계산 뜻이 다르다.

```python
B = np.array([[1, 2], [3, 4]])
C = np.array([[5, 6], [7, 8]])

print(B * C)  # 각 위치의 원소끼리 곱함
print(B @ C)  # 행렬 곱셈: [[19 22], [43 50]]
```

### 왜 크기(shape)가 중요할까?

`(2, 3) @ (3, 4)`는 가능하고 결과는 `(2, 4)`이다. 앞 행렬의 **열 수(3)**와 뒤 행렬의 **행 수(3)**가 같아야 한다.

\[
(m\times n)(n\times p)\longrightarrow(m\times p)
\]

## 4. 행렬을 '변환 규칙'으로 바라보기

다음 행렬은 x좌표를 2배, y좌표를 절반으로 바꾼다.

\[
S=\begin{bmatrix}2&0\\0&0.5\end{bmatrix},\quad
S\begin{bmatrix}3\\2\end{bmatrix}=\begin{bmatrix}6\\1\end{bmatrix}
\]

즉, 행렬은 단순 숫자 표를 넘어 **입력 벡터를 출력 벡터로 바꾸는 함수**로 볼 수 있다.

```python
S = np.array([[2.0, 0.0], [0.0, 0.5]])
point = np.array([3.0, 2.0])
print(S @ point)  # [6. 1.]
```

## 5. 직접 시각화하기

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([3.0, 2.0])
S = np.array([[2.0, 0.0], [0.0, 0.5]])
Sv = S @ v

fig, ax = plt.subplots(figsize=(7, 5))
ax.arrow(0, 0, v[0], v[1], head_width=0.18, length_includes_head=True, label="v")
ax.arrow(0, 0, Sv[0], Sv[1], head_width=0.18, length_includes_head=True, color="tab:orange", label="S @ v")
ax.text(v[0], v[1] + 0.25, "v = (3, 2)")
ax.text(Sv[0] - 1.8, Sv[1] - 0.45, "S @ v = (6, 1)")
ax.set(xlim=(-0.5, 7), ylim=(-0.5, 4), xlabel="x", ylabel="y", title="Vector and matrix transformation")
ax.axhline(0, color="gray", linewidth=0.7)
ax.axvline(0, color="gray", linewidth=0.7)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

관찰할 점: **한 행렬이 벡터의 길이뿐 아니라 방향도 바꿀 수 있다.**

## 6. AI와 연결하기

신경망의 한 층은 보통 다음 형태로 데이터를 바꾼다.

\[
\mathbf{z}=W\mathbf{x}+\mathbf{b}
\]

- `x`: 입력 벡터 (예: 이미지에서 추출한 특징)
- `W`: 가중치 행렬 (학습으로 조정되는 숫자)
- `b`: 편향 벡터
- `z`: 다음 연산에 전달할 결과

예를 들어 입력 특징 3개에서 출력값 2개를 만든다면 `W.shape == (2, 3)`, `x.shape == (3,)`, `b.shape == (2,)`다.

```python
W = np.array([[0.2, 0.5, -0.1], [1.0, 0.0, 0.3]])
x = np.array([2.0, 1.0, 4.0])
b = np.array([0.1, -0.2])

z = W @ x + b
print(z)       # [0.6 3. ]
print(z.shape) # (2,)
```

## 7. 자주 하는 실수

1. `@`와 `*`를 같은 연산으로 생각한다. → **행렬 곱과 원소별 곱은 다르다.**
2. `A.shape`와 `x.shape`를 확인하지 않는다. → 차원 불일치 오류가 난다.
3. 벡터 `(2,)`를 열벡터 `(2, 1)`와 완전히 같다고 생각한다. → 브로드캐스팅 결과가 달라질 수 있다.

## 8. 연습문제

**문제 1.** `v = [6, 8]`의 길이를 손으로 계산한 뒤 `np.linalg.norm`으로 검증해 보자.

**문제 2.** `A = [[2, 1], [0, 3]]`, `x = [4, 2]`일 때 `A @ x`를 계산해 보자.

**문제 3.** `W.shape == (4, 3)`이고 입력 `x.shape == (3,)`라면 출력의 `shape`는 무엇일까?

**문제 4.** `A * B`와 `A @ B`의 결과가 달라지는 2×2 행렬 두 개를 만들어 보자.

<details>
<summary>정답 확인</summary>

1. `sqrt(6² + 8²) = 10`
2. `[10, 6]`
3. `(4,)`
4. 위 3장의 `B`, `C` 예시 참고

</details>

## 9. 완료 체크리스트

- [ ] 벡터와 행렬을 일상 예시로 설명할 수 있다.
- [ ] 행렬 × 벡터 계산을 손으로 풀 수 있다.
- [ ] `@`, `*`, `shape`의 차이를 설명할 수 있다.
- [ ] 그림에서 행렬 변환 전후를 비교할 수 있다.

**다음 학습:** [STEP 1-2. 선형 변환](STEP_1-2_선형_변환.md) — 회전, 스케일링, 전단을 실제 그래프로 확인한다.
