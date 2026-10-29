# STEP 4-5. 학습 과정 시각화 및 비교

> 코디세이 A2-1 AI 수학 · STEP 4 최적화 · Python/NumPy 입문자용

## 학습 목표

- 실험의 공정한 비교 조건을 설정한다.
- 학습률과 Momentum 실험을 한 번에 시각화한다.
- 실험 결과를 표로 요약한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 실험 설계의 기본

**비교하려는 변수 하나만 바꾸는 것**이 핵심이다. 초기 가중치, 데이터, 에포크 수를 동일하게 하며 `lr`이나 `beta`를 하나씩 변경한다.

## 2. 학습 이력을 남기는 코드

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1.,2.,3.]);y=2*x

def experiment(lr,beta=None,epochs=45):
    w=0.;v=0.;data=[]
    for t in range(epochs):
        L=np.mean((w*x-y)**2)
        data.append((t,w,L))
        g=np.mean(2*(w*x-y)*x)
        if beta is None:
            w-=lr*g
        else:
            v=beta*v+g
            w-=lr*v
    return np.array(data)

configs={'GD slow':(.01,None),'GD stable':(.1,None),
         'GD divergent':(.25,None),'Momentum':(.03,.9)}
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,(lr,beta) in configs.items():
    record=experiment(lr,beta)
    axes[0].plot(record[:,0],record[:,1],label=name)
    axes[1].semilogy(record[:,0],record[:,2]+1e-16,label=name)
    print(f'{name:14s} last_logged_MSE={record[-1,2]:.6g}  last_logged_w={record[-1,1]:.6g}')
axes[0].axhline(2,color='gray',linestyle='--')
axes[0].set(xlabel='epoch',ylabel='w',title='Parameter trajectories')
axes[1].set(xlabel='epoch',ylabel='MSE',title='Loss histories (log scale)')
for ax in axes: ax.legend();ax.grid(alpha=.2)
plt.tight_layout();plt.show()
```

## 3. 그래프 해석의 질문

- 어느 실험이 `w=2`에 가장 가까이 갔는가?
- 손실이 단조 감소했는가, 위아래로 진동했는가?
- 발산 곡선의 **y축은 로그 스케일**인가? 이를 표시하지 않으면 독자가 크기를 오해할 수 있다.
- 종료 시점의 손실과 **최저 관측 손실**은 다른 지표다.

## 4. 보고서에 쓸 수 있는 해석 예시

> 동일한 데이터와 시작 가중치에서 학습률 0.1의 GD는 수렴했지만, 0.25는 해당 이차 손실의 이론적 안정 범위(3/14)를 넘어 발산했다. Momentum은 이전 기울기 정보를 누적하므로 빠른 이동과 진동이 모두 나타날 수 있었다.

위 문장은 실제 그래프와 결과를 본 뒤 숫자를 채워 최종 문장으로 작성하자.

---

## 직접 풀어보기

① `loss` 축을 선형 스케일로 바꾸어 비교하라. ② 최종 손실이 최소인 설정을 자동 선택하라. ③ 보고서에 학습률, beta, epochs, 초기 가중치를 표로 적어라.

<details>
<summary>정답·힌트 확인</summary>

① `semilogy`를 `plot`으로 교체. ② `min(configs, key=lambda name: experiment(*configs[name])[-1,2])`와 같이 평가할 수 있다. ③ 비교조건을 동일하게 기록해야 재현 가능하다.

</details>

## 미션 평가 체크

- [ ] 서로 다른 학습 방식·하이퍼파라미터의 손실 그래프, 실험 조건, 결과 해석을 제출할 수 있는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 4-4 Momentum(모멘텀)](STEP_4-4_Momentum모멘텀.md) · [전체 목차](../README.md) · [다음: 5-1 확률 기본 →](../STEP_5_확률_통계/STEP_5-1_확률_기본.md)
