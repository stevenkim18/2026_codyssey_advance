# STEP 6-1. 선형대수 실습 정리

> 코디세이 A2-1 AI 수학 · STEP 6 통합 실습과 제출 · Python/NumPy 입문자용

## 학습 목표

- 회전·스케일링·전단 그래프를 한 번에 출력한다.
- 행렬식 면적비, Power Iteration, SVD를 자동 검증한다.
- 평가자가 실행할 수 있는 하나의 스크립트를 구성한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 이번 단원은 무엇을 제출하는가?

STEP 1에서 배운 이론을 **평가용 재현 코드**로 묶는다. 평가자가 프로그램을 실행하면 다음이 보여야 한다.

| 항목 | 충족 조건 |
|---|---|
| 선형 변환 | 회전·스케일링·전단 **3종 그래프** |
| 행렬식 | `det(A)`로 얻은 면적비와 실제 다각형 면적비 오차 **1% 이내** |
| Power Iteration | `np.linalg.eig`의 기준 고유값과 **상대오차 5% 이내** |
| 이미지 SVD | `k=10,50,100`을 **한 Figure에 비교** |

## 2. 별도 파일 하나로 실행할 수 있는 통합 실습

아래 전체 코드를 `step6_1_linear_algebra.py`로 저장하여 실행한다. 외부 이미지 파일 없이 NumPy로 이미지 데이터를 만든다.

```python
import numpy as np
import matplotlib.pyplot as plt

# ── A. 선형 변환 3종 ──
square=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]],dtype=float)
angle=np.deg2rad(35)
rotation=np.array([[np.cos(angle),-np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])
scale=np.array([[1.8,0],[0,.75]])
shear=np.array([[1.,.7],[0,1.]])
transforms=[('Rotation',rotation),('Scaling',scale),('Shear',shear)]
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,(name,A) in zip(axes,transforms):
    result=square@A.T
    ax.plot(square[:,0],square[:,1], 'k--',label='Original')
    ax.plot(result[:,0],result[:,1],'-o',label='Transformed')
    ax.set_title(name); ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-1.5,3); ax.set_ylim(-1.5,3); ax.grid(alpha=.2)
axes[0].legend(); plt.tight_layout();plt.show()

# ── B. det(A) vs 실제 면적비 (다각형 shoelace 공식을 사용) ──
def polygon_area(points):
    x,y=points[:,0],points[:,1]
    return .5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

base=square[:-1]
A=np.array([[2.,1.],[0.,2.5]])  # det=5
ratio=polygon_area(base@A.T)/polygon_area(base)
expected=abs(np.linalg.det(A))
relative_error=abs(ratio-expected)/max(expected,1e-12)
print(f'[AREA] observed={ratio:.6f}, abs(det)={expected:.6f}, relative_error={relative_error:.3%}')
assert relative_error < .01

# ── C. Power Iteration vs eig ──
B=np.array([[4.,1.],[1.,3.]]) # 양의 고유값을 갖는 대칭행렬
v=np.array([1.,1.]);v=v/np.linalg.norm(v)
for _ in range(50):
    v=B@v;v=v/np.linalg.norm(v)
power_eigenvalue=float(v@B@v)  # Rayleigh quotient
ref_eigenvalue=float(np.max(np.linalg.eigvalsh(B)))
eigen_error=abs(power_eigenvalue-ref_eigenvalue)/abs(ref_eigenvalue)
print(f'[EIGEN] power={power_eigenvalue:.8f}, eig={ref_eigenvalue:.8f}, error={eigen_error:.3%}')
assert eigen_error < .05

# ── D. SVD 이미지 압축/복원 ──
rng=np.random.default_rng(42)
h,w=160,160
y,x=np.mgrid[0:h,0:w]
image=np.clip(.25+.15*np.sin(x/14)+.18*np.cos(y/19)
              +.42*np.exp(-((x-60)**2+(y-70)**2)/(2*24**2))
              +.07*rng.normal(size=(h,w)),0,1)
U,s,Vt=np.linalg.svd(image,full_matrices=False)
def reconstruct(k):
    return (U[:,:k]*s[:k])@Vt[:k,:]
fig,axes=plt.subplots(1,4,figsize=(14,4))
axes[0].imshow(image,cmap='gray',vmin=0,vmax=1);axes[0].set_title('Original')
for ax,k in zip(axes[1:],[10,50,100]):
    approx=reconstruct(k)
    err=np.mean((image-approx)**2)
    print(f'[SVD] k={k:3d}, MSE={err:.8f}')
    ax.imshow(np.clip(approx,0,1),cmap='gray',vmin=0,vmax=1)
    ax.set_title(f'k={k}, MSE={err:.4f}')
for ax in axes:ax.axis('off')
plt.suptitle('SVD compressed reconstruction')
plt.tight_layout();plt.show()
```

## 3. 반드시 설명할 포인트

- 회전·스케일링·전단은 서로 다른 **행렬 변환**이다.
- 면적의 배율은 부호가 아니라 `|det(A)|`다(행렬식의 음수는 방향 반전을 포함한다).
- Power Iteration은 일반적으로 **절댓값이 가장 큰 고유값** 성분으로 수렴한다(우세한 고유값, 시작벡터 등의 조건 필요). 이 실습에서는 대칭 양의 정부호 행렬을 사용해 `eigvalsh`의 가장 큰 고유값과 비교했다.
- SVD의 `k`는 **남기는 특이값 개수**이며 일반적으로 k가 커지면 근사 오차가 줄어든다.

## 4. 평가자가 물어볼 만한 질문

Q. 면적비를 `det(A)`와 비교하는 이유는?  A. 선형 변환은 2차원 면적을 `|det(A)|`배 바꾸기 때문이다.

Q. `k=100`이 `k=10`보다 원본에 가까운 이유는? A. 더 많은 특이 성분을 포함하기 때문이다.

---

## 직접 풀어보기

① 회전행렬의 행렬식이 1인지 확인하라. ② 전단행렬의 행렬식이 1이라는 의미를 설명하라. ③ SVD k=10·50·100의 MSE가 비증가하는지 확인하라.

<details>
<summary>정답·힌트 확인</summary>

① `np.linalg.det(rotation)≈1`. ② 전단은 단위 면적을 유지하며 형태를 기울인다. ③ `MSE10≥MSE50≥MSE100`이 수치 오차 범위에서 성립해야 한다.

</details>

## 미션 평가 체크

- [ ] 선형대수 평가 4개를 실행 후 출력/화면/캡처로 증빙할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 5-6 확률적 손실함수 이해(NLL 연결)](../STEP_5_확률_통계/STEP_5-6_확률적_손실함수_이해NLL_연결.md) · [전체 목차](../README.md) · [다음: 6-2 수치 미분·역전파 검증 →](STEP_6-2_수치_미분·역전파_검증.md)
