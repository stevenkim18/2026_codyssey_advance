# STEP 1-5. 특이값 분해(SVD): 중요한 정보만 남겨 이미지 복원하기

> A2-1 AI 수학 · STEP 1 선형대수 · 난이도: 초급~중급 · 권장 학습 시간: 120~150분

## 0. 이번 학습 목표

- SVD를 **행렬을 여러 중요도 단계로 나누는 방법**으로 이해한다.
- `A = U @ Σ @ Vᵀ`의 의미를 설명한다.
- NumPy의 `np.linalg.svd`로 행렬을 분해하고 복원한다.
- **k=10, 50, 100**일 때 이미지 복원 결과를 **하나의 Figure**에 비교한다.
- 복원 오차(MSE)와 저장 비용의 변화까지 확인한다.

```bash
python -m pip install numpy matplotlib
```

---

## 1. 직관: 사진을 특징의 층으로 나누기

사진은 수많은 픽셀 값으로 이루어져 있다. 흑백 이미지가 세로 160픽셀, 가로 160픽셀이라면 **160×160 숫자 행렬**로 생각할 수 있다.

그런데 사진의 모든 숫자가 똑같이 중요하지는 않다. 밝고 어두운 큰 구조가 먼저 보이고, 세밀한 질감이나 잡음은 나중에 드러날 수 있다.

**SVD(Singular Value Decomposition, 특이값 분해)**는 행렬을 구성하는 패턴들을 나누고, 중요한 패턴부터 남길 수 있게 해 준다.

비유하자면 **사진을 레고 층으로 분해하고 중요한 층부터 다시 쌓는 것**이다.

## 2. SVD 공식 읽기

\[
A=U\Sigma V^T
\]

- `A`: 원본 행렬 또는 흑백 이미지
- `U`: 왼쪽 방향 정보를 담은 직교 벡터들
- `Σ`(시그마): 각 패턴의 중요도를 나타내는 **특이값**을 대각에 배치한 행렬
- `Vᵀ`: 오른쪽 방향 정보를 담은 직교 벡터들의 전치행렬

행렬 크기가 `m × n`인 경우, 축약 SVD에서 `r=min(m,n)`이라 하면:

| 구성 | shape |
|---|---|
| `A` | `(m, n)` |
| `U` | `(m, r)` |
| `s` (대각 원소만 보관) | `(r,)` |
| `Vt` | `(r, n)` |

NumPy의 `np.linalg.svd(A, full_matrices=False)`는 대각행렬 `Σ`를 직접 만들지 않고 **특이값 1차원 배열 `s`**를 반환한다.

## 3. 작은 행렬로 직접 이해하기

\[
A=\begin{bmatrix}3&0\\0&1\end{bmatrix}
\]

이 행렬은 x방향을 3배, y방향을 1배 한다. 특이값은 `3`, `1`이다. 가장 큰 특이값 `3`만 남기면

\[
A_{k=1}=\begin{bmatrix}3&0\\0&0\end{bmatrix}
\]

이 된다. 두 번째 방향의 정보는 버리고 큰 패턴 하나만 남긴 것이다.

```python
import numpy as np

A = np.array([[3.0, 0.0], [0.0, 1.0]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)

full = (U * s) @ Vt          # 모든 특이값으로 복원
k1 = (U[:, :1] * s[:1]) @ Vt[:1, :]  # 가장 큰 특이값만 사용

print("특이값:", s)           # [3. 1.]
print("원본과 일치:", np.allclose(A, full))  # True
print("k=1 근사:\n", np.round(k1, 6))
```

결과:

```text
특이값: [3. 1.]
원본과 일치: True
k=1 근사:
 [[3. 0.]
  [0. 0.]]
```

> 특이값은 **0 이상**이며 NumPy는 기본적으로 큰 값부터 정렬해 준다.

## 4. k개만 골라 복원하는 수식

\[
A_k=\sum_{i=1}^{k}\sigma_i\mathbf u_i\mathbf v_i^T
\]

`k`는 **남겨 둘 특이값(패턴)의 개수**다.

- `k`가 작다 → 중요한 패턴 몇 개만 사용 → 세부 정보가 많이 사라질 수 있음
- `k`가 크다 → 더 많은 패턴 사용 → 원본에 가까움
- `k = min(m, n)` → 부동소수점 오차를 제외하면 원본 복원

```python
# U: (m, r), s: (r,), Vt: (r, n)
k = 1
reconstructed = (U[:, :k] * s[:k]) @ Vt[:k, :]
```

`U[:, :k] * s[:k]`는 NumPy의 **브로드캐스팅**을 이용하여 `U`의 각 열에 대응하는 특이값을 곱한다. `*`가 원소별 계산이지만, `s`의 크기를 열에 맞춰 확장하므로 의도한 연산이 된다.

## 5. A2-1 핵심 실습: k=10, 50, 100 이미지 비교

외부 이미지 다운로드나 추가 라이브러리 없이 실행할 수 있도록 **NumPy로 흑백 테스트 이미지**를 만든다. 실제 사진 파일이 있다면 뒤의 7장에서 교체하는 방법을 참고하자.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 가상의 흑백 이미지를 만든다. 크기는 160×160.
height, width = 160, 160
y, x = np.mgrid[0:height, 0:width]
rng = np.random.default_rng(42)

# 부드러운 배경, 밝은 영역, 수직 무늬, 약간의 세부 잡음
background = 0.24 + 0.15 * np.sin(x / 13.0) + 0.13 * np.cos(y / 17.0)
spot = 0.50 * np.exp(-((x - 55) ** 2 + (y - 75) ** 2) / (2 * 22 ** 2))
stripe = 0.18 * np.sin(x / 3.0) * ((x > 90) & (y < 110))
noise = 0.09 * rng.normal(size=(height, width))
image = np.clip(background + spot + stripe + noise, 0.0, 1.0)

# 2. SVD 분해. 이미지 = U @ diag(s) @ Vt
U, s, Vt = np.linalg.svd(image, full_matrices=False)

# 3. k개 특이값만 남겨 복원한다.
def reconstruct(k):
    if not (1 <= k <= len(s)):
        raise ValueError(f"k는 1부터 {len(s)} 사이여야 합니다.")
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

# 4. 원본과 k=10, 50, 100을 한 Figure에서 비교한다.
k_values = [10, 50, 100]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")

for ax, k in zip(axes[1:], k_values):
    restored = reconstruct(k)
    mse = float(np.mean((image - restored) ** 2))

    # 표시할 때만 [0,1] 범위로 제한: MSE 계산에는 원래 복원값 사용
    ax.imshow(np.clip(restored, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"k={k}\nMSE={mse:.5f}")
    ax.axis("off")
    print(f"k={k:3d}, MSE={mse:.8f}")

plt.suptitle("SVD Image Reconstruction: k = 10, 50, 100")
plt.tight_layout()
plt.show()

# 5. 오차가 점점 줄어드는지 추가로 확인
errors = [np.mean((image - reconstruct(k)) ** 2) for k in k_values]
print("k 증가에 따른 MSE 감소:", all(a >= b for a, b in zip(errors, errors[1:])))
```

### 그래프에서 확인할 사항

- 하나의 Figure에서 **원본과 k=10, 50, 100** 이미지를 한눈에 비교한다.
- `k=10`: 전체적인 모양이 남고 잡음 같은 작은 특징은 덜 보인다.
- `k=50`: 좀 더 세부적인 패턴이 살아난다.
- `k=100`: 원본과 더욱 가까워진다.
- 일반적으로 `k`가 증가하면 **최적 랭크-k 근사 오차(MSE)가 증가하지 않는다.**

여기서 이미지 픽셀은 `0`(검정)에서 `1`(흰색) 범위를 사용한다.

## 6. 이것이 정말 '압축'일까?

원본 흑백 이미지에 있는 수치의 개수는 `m × n`이다. 축약 SVD로 `k`개의 패턴을 저장할 때 필요한 수치는 대략 다음과 같다.

\[
\text{저장 숫자 수}=mk+k+nk=k(m+n+1)
\]

위 160×160 예시에서:

| k | 저장 숫자 수 | 원본 숫자 수 | 단순 수치 개수 비율 |
|---:|---:|---:|---:|
| 10 | 3,210 | 25,600 | 약 12.5% |
| 50 | 16,050 | 25,600 | 약 62.7% |
| 100 | 32,100 | 25,600 | 약 125.4% |

**중요:** `k=100`은 시각적으로 원본에 더 가깝지만, 이 작은 이미지에서는 **수치 개수 기준으로는 오히려 더 많이 저장**할 수 있다. 따라서 `k`가 크다고 무조건 압축 효율이 좋은 것은 아니다.

또한 이것은 수치 **개수**만 비교한 값이다. 실제 PNG/JPEG 파일 크기는 인코딩, 픽셀 데이터 타입, 저장 형식 등에 따라 달라져 이 표로 직접 예측할 수 없다.

```python
m, n = image.shape
for k in [10, 50, 100]:
    count = k * (m + n + 1)
    ratio = count / (m * n) * 100
    print(f"k={k:3d}: 원본 대비 저장 수치 수 = {ratio:.1f}%")
```

## 7. 실제 이미지 파일을 이용하고 싶다면 (선택)

이미지 파일이 있다면 다음 코드로 **흑백 행렬**로 바꿀 수 있다. Pillow를 추가 설치해야 한다.

```bash
python -m pip install pillow
```

```python
from PIL import Image
import numpy as np

# image.jpg를 실행 파일과 같은 폴더에 둔다.
# 최소 한 변이 100픽셀 이상이어야 k=100을 그대로 사용할 수 있다.
with Image.open("image.jpg") as img:
    gray = img.convert("L").resize((160, 160))
    image = np.asarray(gray, dtype=float) / 255.0

U, s, Vt = np.linalg.svd(image, full_matrices=False)
# 위 5장의 reconstruct() 함수와 시각화 코드를 그대로 활용한다.
```

주의: `k`는 `min(height, width)` 이하여야 한다. 예를 들어 이미지가 64×64이면 `k=100`은 불가능하다. 평가에서 실제로 `k=10, 50, 100`을 구분해 비교하려면 **최소 크기 100×100**인 이미지를 사용하자. 이 문서의 기본 예제는 160×160이므로 문제가 없다.

## 8. 고유값 분해와 SVD는 무엇이 다를까?

| 구분 | 고유값 분해 | SVD |
|---|---|---|
| 대표 식 | `Av = λv` | `A = UΣVᵀ` |
| 대상 | 고유값은 정사각행렬에서 정의됨 | 직사각형 행렬에도 적용 가능 |
| 중요한 값 | 고유값 λ | 특이값 σ (0 이상) |
| 대표 활용 | 변환의 고유방향, PCA | 이미지 압축, 노이즈 완화, 저랭크 근사 |

**연결되는 부분:** 행렬 `A`의 특이값 제곱은 `A.T @ A`의 고유값과 관계가 있다. 하지만 처음에는 **SVD는 직사각형 행렬도 분해하고, 큰 특이값부터 남기며 복원한다**는 개념만 이해해도 충분하다.

## 9. A2-1 평가에 대응하는 체크리스트

평가 항목: **SVD 압축에서 k=10, 50, 100 복원 결과가 하나의 Figure에 비교 출력되어 있는가?**

- [ ] `np.linalg.svd(image, full_matrices=False)`를 호출했다.
- [ ] `reconstruct(k)` 함수로 복원했다.
- [ ] `k=10, 50, 100`을 각각 계산했다.
- [ ] 한 Figure의 서브플롯에 세 복원 이미지가 함께 표시된다.
- [ ] 원본도 표시해 비교하기 쉽다.
- [ ] `k`별 MSE를 출력하고 차이를 설명할 수 있다.

## 10. 연습문제

**문제 1.** `k=1`로 복원하면 `k=100`보다 원본에 가깝거나 멀까? 이유를 설명해 보자.

**문제 2.** 이미지가 `(80, 120)`이면 사용할 수 있는 최대 `k`는 얼마일까?

**문제 3.** 5장의 코드를 수정해 `k=[5, 20, 80]`으로 비교해 보자.

**문제 4.** `k=10` 결과에 비해 `k=50` 결과의 MSE가 작은지 코드로 검증해 보자.

**문제 5.** 이미지가 `(160, 160)`일 때 `k=100`의 수치 개수 비율이 100%를 넘는 이유는 무엇일까?

<details>
<summary>정답 확인</summary>

1. `k=1`은 보통 더 멀다. 중요한 패턴 1개만 남기므로 정보 손실이 크다.
2. `min(80, 120)=80`
3. 직접 실행해 세부 정보의 차이를 비교해 보자.
4. `np.mean((image-reconstruct(50))**2) <= np.mean((image-reconstruct(10))**2)`가 `True`여야 한다(부동소수점 오차 가능).
5. 저장하는 `U`, `s`, `Vt`의 수치 총합이 원본 픽셀 수보다 많기 때문이다.

</details>

## 11. STEP 1 전체 복습

| 학습 단원 | 핵심 질문 | 대표 NumPy API | 평가 연결 |
|---|---|---|---|
| 1-1 벡터·행렬 | 숫자 묶음에 규칙을 어떻게 적용할까? | `@`, `.shape` | 기본 연산 설명 |
| 1-2 선형 변환 | 도형이 어떻게 움직일까? | `A @ points` | 회전·스케일링·전단 3종 |
| 1-3 행렬식 | 면적이 몇 배가 될까? | `np.linalg.det` | 면적비 오차 1% 이내 |
| 1-4 고유값·고유벡터 | 변해도 방향이 남는 벡터는? | `np.linalg.eig` | Power Iteration 오차 5% 이내 |
| 1-5 SVD | 중요한 정보만 남길 수 있을까? | `np.linalg.svd` | k=10/50/100 한 Figure |

**STEP 1을 마쳤다면:** 선형대수 평가 항목을 스스로 구현할 수 있는 기초가 갖춰졌다. 다음은 **STEP 2 미적분**에서 수치미분과 Gradient를 학습한다.
