# STEP 1-4. 고유값과 고유벡터: 방향이 그대로인 특별한 화살표

> A2-1 AI 수학 · STEP 1 선형대수 · 난이도: 초급~중급 · 권장 학습 시간: 120~150분

## 0. 이 수업의 목표

- `A @ v = λ * v`를 **방향과 크기**의 관점에서 설명한다.
- 고유값(λ)과 고유벡터(v)를 직접 검증한다.
- **Power Iteration(거듭제곱 반복법)**을 NumPy로 구현한다.
- `np.linalg.eig`와 비교해 고유값 오차 **5% 이내**인지 출력한다.
- 왜 반복할수록 특정 방향이 남는지 직관적으로 이해한다.

```bash
python -m pip install numpy matplotlib
```

---

## 1. 직관: 종이를 늘려도 방향이 그대로인 선

종이에 여러 방향의 화살표를 그려 두고 종이를 가로로 크게 늘린다고 상상해 보자. 대부분 화살표는 방향이 바뀌지만, **가로 방향 화살표**는 여전히 가로 방향을 향한다.

이처럼 행렬로 변환해도 **벡터의 방향(또는 반대 방향)을 따라 배율만 달라지는 특별한 벡터**를 *고유벡터*라고 한다. 그 배율을 *고유값*이라고 한다.

\[
A\mathbf v=\lambda\mathbf v
\]

- `A`: 변환 행렬
- `v`: 고유벡터 (`v ≠ 0`)
- `λ`: 고유값 (몇 배로 늘거나 줄어드는가)

> `λ`가 음수이면 화살표 방향이 반대로 뒤집힌다. `λ=0`이면 그 방향이 0벡터로 눌린다.

## 2. 제일 쉬운 예제: 대각행렬

\[
A=\begin{bmatrix}3&0\\0&1\end{bmatrix}
\]

오른쪽을 향한 `v1 = [1,0]`과 위쪽을 향한 `v2 = [0,1]`을 넣어 보자.

\[
A\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}3\\0\end{bmatrix}
=3\begin{bmatrix}1\\0\end{bmatrix}
\]

\[
A\begin{bmatrix}0\\1\end{bmatrix}
=\begin{bmatrix}0\\1\end{bmatrix}
=1\begin{bmatrix}0\\1\end{bmatrix}
\]

따라서 고유값은 각각 `3`, `1`이고, 고유벡터 방향은 x축과 y축이다.

```python
import numpy as np

A = np.array([[3.0, 0.0], [0.0, 1.0]])
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 1.0])

print(np.allclose(A @ v1, 3 * v1))  # True
print(np.allclose(A @ v2, 1 * v2))  # True
```

## 3. 조금 더 현실적인 예제

이번에는 다음 대칭행렬을 쓰자.

\[
A=\begin{bmatrix}4&1\\1&3\end{bmatrix}
\]

고유값은 약 `4.618034`와 `2.381966`이다. 가장 큰 고유값에 해당하는 단위 고유벡터는 대략 다음과 같다(부호는 반대여도 같다).

\[
\mathbf v\approx\begin{bmatrix}0.850651\\0.525731\end{bmatrix}
\]

확인하면

\[
A\mathbf v\approx4.618034\mathbf v
\]

이다. 계산의 양쪽이 거의 같다는 것이 핵심이다.

```python
import numpy as np

A = np.array([[4.0, 1.0], [1.0, 3.0]])
values, vectors = np.linalg.eig(A)

idx = np.argmax(np.abs(values))  # 절댓값이 가장 큰 고유값
lam = float(values[idx])
v = vectors[:, idx]

print("고유값:", round(lam, 6))
print("고유벡터:", np.round(v, 6))
print("Av = λv 검증:", np.allclose(A @ v, lam * v))
```

**주의:** 고유벡터 `v`와 `-v`는 같은 고유벡터 방향을 뜻한다. 부호가 예제와 달라도 오답이 아니다.

## 4. Power Iteration: 행렬을 반복해서 곱하기

고유값을 처음부터 수식으로 풀지 않고, **행렬을 여러 번 곱해 가장 강한 방향을 찾는** 방법이다.

시작 벡터가 여러 고유벡터 방향의 합으로 표현된다고 생각해 보자. 반복해서 `A`를 곱하면 절댓값이 큰 고유값 방향의 성분이 상대적으로 더 빨리 커진다. 매번 길이를 1로 만들어 주면 숫자가 무한정 커지는 것을 막고 **방향만 추적**할 수 있다.

반복식은 다음과 같다.

\[
\mathbf w=A\mathbf v,\qquad
\mathbf v_{new}=\frac{\mathbf w}{\|\mathbf w\|}
\]

고유값 추정에는 **레이리 몫(Rayleigh quotient)**을 사용한다.

\[
\hat\lambda=\frac{\mathbf v^T A\mathbf v}{\mathbf v^T\mathbf v}
\]

길이 1의 벡터라면 분모가 1이므로 `v @ A @ v`로 계산할 수 있다.

### 직접 구현

```python
import numpy as np

A = np.array([[4.0, 1.0], [1.0, 3.0]])

def power_iteration(A, max_iter=1000, tol=1e-10):
    # 1) 시작 벡터. 최대 고유벡터 성분이 0이 아닌 것으로 선택
    v = np.ones(A.shape[0], dtype=float)
    v = v / np.linalg.norm(v)

    for iteration in range(1, max_iter + 1):
        # 2) 행렬을 곱하고 길이를 1로 정규화
        w = A @ v
        norm_w = np.linalg.norm(w)
        if np.isclose(norm_w, 0.0):
            raise ValueError("Av가 영벡터여서 정규화할 수 없습니다.")
        v_next = w / norm_w

        # 3) 수렴 확인: 고유벡터의 부호가 바뀌어도 같은 방향
        difference = min(np.linalg.norm(v_next - v),
                         np.linalg.norm(v_next + v))
        v = v_next
        if difference < tol:
            break

    # 4) 레일리 몫으로 고유값 추정
    eigenvalue = float((v @ A @ v) / (v @ v))
    return eigenvalue, v, iteration

estimated_lambda, estimated_v, iterations = power_iteration(A)

print("반복 횟수:", iterations)
print("추정 고유값:", round(estimated_lambda, 6))
print("추정 고유벡터:", np.round(estimated_v, 6))
print("Av - λv 크기:", np.linalg.norm(A @ estimated_v - estimated_lambda * estimated_v))
```

> 이 반복법은 보통 **절댓값이 가장 큰 고유값이 유일**하고 초기 벡터에 해당 고유벡터 성분이 있을 때 잘 작동한다. 모든 행렬에서 항상 성공하는 알고리즘은 아니다.

## 5. `np.linalg.eig`와 오차 비교하기

평가에서는 자체 구현 결과를 라이브러리 결과와 비교해야 한다.

```python
import numpy as np

# 앞 장의 A와 power_iteration 함수가 정의된 상태에서 실행한다.
actual_values, actual_vectors = np.linalg.eig(A)
idx = np.argmax(np.abs(actual_values))
actual_lambda = float(actual_values[idx])

relative_error = abs(estimated_lambda - actual_lambda) / abs(actual_lambda) * 100
print(f"Power Iteration: {estimated_lambda:.8f}")
print(f"np.linalg.eig:  {actual_lambda:.8f}")
print(f"고유값 상대 오차: {relative_error:.8f}%")
print("5% 이내:", relative_error <= 5.0)
```

**예상:** 두 결과 모두 `4.61803399` 근처이고, 오차는 거의 `0%`이다.

검증을 더 엄밀하게 하려면 `np.linalg.norm(A @ estimated_v - estimated_lambda * estimated_v)`이 매우 작은지도 확인하자.

## 6. 그림으로 고유벡터 확인하기

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[4.0, 1.0], [1.0, 3.0]])
values, vectors = np.linalg.eig(A)
idx = np.argmax(np.abs(values))
v = vectors[:, idx]
Av = A @ v

fig, ax = plt.subplots(figsize=(6, 6))
ax.arrow(0, 0, v[0], v[1], color="tab:blue", head_width=0.13,
         length_includes_head=True)
ax.arrow(0, 0, Av[0], Av[1], color="tab:orange", head_width=0.13,
         length_includes_head=True)
ax.text(v[0] + 0.1, v[1] + 0.1, "v", color="tab:blue")
ax.text(Av[0] + 0.1, Av[1] + 0.1, "A @ v", color="tab:orange")
ax.axhline(0, color="gray", linewidth=0.6)
ax.axvline(0, color="gray", linewidth=0.6)
ax.set(xlim=(-1, 5), ylim=(-1, 5), title="Eigenvector stays on the same line")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
```

**관찰:** 파란 화살표와 주황 화살표가 같은 직선 위에 있다. 길이는 달라도 방향은 유지된다.

## 7. 실전에서 자주 틀리는 부분

| 오해 | 올바른 이해 |
|---|---|
| 모든 벡터가 고유벡터다 | 특별한 방향의 벡터만 해당한다 |
| 고유값이 무조건 양수다 | 음수·0·복소수도 가능하다 |
| 고유벡터의 부호가 다르면 오답이다 | `v`와 `-v`는 같은 고유방향이다 |
| Power Iteration은 언제나 가장 큰 양수 고유값을 찾는다 | 기본적으로 **절댓값이 가장 큰** 고유값을 목표로 한다 |
| 반복 횟수만 늘리면 무조건 해결된다 | 수렴 조건이 맞지 않으면 실패할 수 있다 |

## 8. AI에서는 어디에 사용할까?

고유값과 고유벡터는 데이터의 **주된 변화 방향**을 찾는 방법과 연결된다. 대표적으로 주성분 분석(PCA)은 데이터의 공분산 행렬에 있는 주요 고유벡터 방향을 사용한다. 이미지·텍스트에서 특징을 압축하거나 데이터 분포를 이해할 때 기초가 된다.

## 9. 연습문제

**문제 1.** `A = diag(5, 2)`에서 Power Iteration을 반복하면 어느 방향으로 수렴할까?

**문제 2.** `v=[1, 0]`, `A=diag(-3, 1)`이면 `v`의 고유값은?

**문제 3.** `max_iter=3`과 `max_iter=1000`의 오차를 비교해 보자. (단, 작은 반복 횟수에서도 `tol`을 지나치게 작게 설정하지 않은 경우)

**문제 4.** 초기 벡터를 `[0, 1]`로 바꾸면 `A=diag(5, 2)`에서 왜 고유값 5를 찾지 못할까?

<details>
<summary>정답 확인</summary>

1. 보통 `[1, 0]` 방향, 고유값 `5`
2. `-3` (방향은 반대로 뒤집힘)
3. 대체로 반복 횟수가 늘면 오차가 작아진다. 실제 결과를 출력해 확인하자.
4. 첫 번째 고유벡터 성분이 시작부터 0이어서 반복 곱으로 새롭게 생기지 않는다.

</details>

## 10. A2-1 평가 체크리스트

- [ ] `A=[[4,1],[1,3]]`을 사용해 Power Iteration을 구현했다.
- [ ] 추정 고유값과 벡터를 출력했다.
- [ ] `np.linalg.eig` 결과와 상대 오차를 출력했다.
- [ ] 상대 오차가 5% 이내임을 출력했다.
- [ ] `Av ≈ λv`를 검증하고 오차가 작은 이유를 설명할 수 있다.

**다음 학습:** [STEP 1-5. 특이값 분해(SVD)](STEP_1-5_특이값_분해_SVD.md) — 행렬을 중요한 정보부터 남기며 근사한다.
