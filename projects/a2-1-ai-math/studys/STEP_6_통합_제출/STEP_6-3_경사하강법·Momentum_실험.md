# STEP 6-3. 경사하강법·Momentum 실험

> 코디세이 A2-1 AI 수학 · STEP 6 통합 실습과 제출 · Python/NumPy 입문자용

## 학습 목표

- 같은 초기조건으로 Vanilla GD와 Momentum을 비교한다.
- 학습률을 바꿔 수렴/발산을 확인한다.
- 실험 결과 그래프를 파일로 저장한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 이전 단원의 공식 복습

회귀 `ŷ=wx`, `L=mean((ŷ-y)²)`에서

\[\frac{dL}{dw}=\operatorname{mean}[2(wx-y)x]\]

GD는 `w-=αg`, Momentum은 `v=βv+g`, `w-=αv`다. 두 알고리즘이 사용하는 **Gradient는 같다**.

## 2. 실행 및 저장이 가능한 비교 코드

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

outdir=Path('outputs');outdir.mkdir(exist_ok=True)
x=np.array([1.,2.,3.]);y=2*x

def run(lr,beta=None,epochs=50):
    w=0.;v=0.;data=[]
    for epoch in range(epochs):
        prediction=w*x
        loss=np.mean((prediction-y)**2)
        gradient=np.mean(2*(prediction-y)*x)
        data.append((epoch,w,loss))
        if beta is None:
            w-=lr*gradient
        else:
            v=beta*v+gradient
            w-=lr*v
    return np.array(data)

configs={
    'GD slow (0.01)':(.01,None),
    'GD stable (0.10)':(.1,None),
    'GD divergent (0.25)':(.25,None),
    'Momentum (0.03, 0.9)':(.03,.9),
}
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,(lr,beta) in configs.items():
    data=run(lr,beta)
    axes[0].plot(data[:,0],data[:,1],label=name)
    axes[1].semilogy(data[:,0],data[:,2]+1e-16,label=name)
    print(f'{name}: epoch={int(data[-1,0])} w={data[-1,1]:.6g} MSE={data[-1,2]:.6g}')
axes[0].axhline(2,color='gray',linestyle='--')
axes[0].set(xlabel='epoch',ylabel='weight',title='Weight vs epoch')
axes[1].set(xlabel='epoch',ylabel='MSE (log)',title='Loss vs epoch')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(outdir/'optimization_comparison.png',dpi=160)
plt.show()
print('그래프 저장 위치:',outdir/'optimization_comparison.png')
```

## 3. 그래프에 적어야 할 해석

- `lr=.01`: 안정적이지만 느린 편이다.
- `lr=.1`: 이 예제의 이론적 수렴 조건 `0<α<3/14` 안에 있다.
- `lr=.25`: 수렴 조건 바깥에 있으므로 MSE가 커진다.
- Momentum: 과거 기울기를 누적하므로 과도하게 이동하는 구간이 있을 수 있다.

한 번의 학습 실험에서 성능이 높게 나왔다고 해서 **모든 문제에서 해당 설정이 최적**이라고 일반화하면 안 된다.

## 4. 평가 발표 예시

> 같은 데이터(`x=[1,2,3]`, `y=2x`), 초기 가중치 0, 50회 업데이트를 유지하면서 학습률과 Momentum 여부만 바꿨습니다. 손실 그래프에서 수렴/발산을 확인했고, 변화 이유를 Gradient 업데이트 공식으로 설명할 수 있습니다.

---

## 직접 풀어보기

① `lr=.2`로 추가 실험하라. ② Momentum `beta=0`이 Vanilla GD와 같음을 검증하라(학습률을 동일하게 설정할 것). ③ 그래프 PNG를 README에 삽입하라.

<details>
<summary>정답·힌트 확인</summary>

① 수렴 조건 안이지만 진동 가능. ② 예: `np.allclose(run(.03,0),run(.03,None))`는 True. ③ `![최적화 비교](outputs/optimization_comparison.png)`.

</details>

## 미션 평가 체크

- [ ] 동일 조건 비교, 수렴·발산 설명, 그래프 PNG 저장을 마쳤는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 6-2 수치 미분·역전파 검증](STEP_6-2_수치_미분·역전파_검증.md) · [전체 목차](../README.md) · [다음: 6-4 결과 그래프·해석 정리 →](STEP_6-4_결과_그래프·해석_정리.md)
