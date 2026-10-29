# STEP 6-4. 결과 그래프·해석 정리

> 코디세이 A2-1 AI 수학 · STEP 6 통합 실습과 제출 · Python/NumPy 입문자용

## 학습 목표

- 그래프와 출력 값을 평가 항목에 맞춰 묶는다.
- 오차·축·범례 등 재현 가능한 시각화 표기를 적용한다.
- 실험 결론과 한계를 구분해 작성한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 평가자는 무엇을 보고 통과 여부를 판단할까?

다른 사람이 프로젝트를 열고 **즉시 결과와 검증 수치**를 찾을 수 있어야 한다. 멋진 색깔보다 중요한 것은 제목, 축의 의미, 단위, 범례, 비교 조건, 검증 기준이다.

## 2. 결과 요약표 템플릿

| 범주 | 출력/이미지 | 기준 | 해석 |
|---|---|---|---|
| 선형 변환 | 3종 변환 그래프 | 3종 정상 출력 | 회전/스케일/전단 형태 비교 |
| 면적 | 실제 면적비, `|det(A)|`, 오차 | 상대오차 1% 이내 | 행렬식의 면적 배율 의미 |
| 고유값 | Power Iteration vs eig | 상대오차 5% 이내 | 수렴값의 의미 |
| SVD | 원본, k=10/50/100 | 한 Figure | k 증가 시 복원 정확도 |
| 미분 | 수치/해석/오차 | 절대오차 `1e-4` 이내 | 차분 근사의 정확성 |
| Gradient | 등고선과 화살표 | 방향 수직 | 최급상승 방향 |
| 역전파 | 순전파, BCE, 기울기 | 손계산/코드 비교 | 연쇄법칙 적용 |
| 최적화 | GD/Momentum 그래프 | 수렴/발산 비교 | 학습률·관성 영향 |

## 3. 자동 결과 파일 만들기

아래 예시는 독립적으로 실행 가능하며 그래프 PNG, 요약 CSV를 같은 폴더에 만든다.

```python
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

output=Path('outputs');output.mkdir(exist_ok=True)
rows=[]
x=np.array([1.,2.,3.]);y=2*x
for name,lr in [('small',.01),('stable',.1),('divergent',.25)]:
    w=0.
    for epoch in range(35):
        loss=np.mean((w*x-y)**2)
        rows.append({'experiment':name,'lr':lr,'epoch':epoch,'w':w,'mse':loss})
        g=np.mean(2*(w*x-y)*x)
        w-=lr*g

with (output/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['experiment','lr','epoch','w','mse'])
    writer.writeheader();writer.writerows(rows)

for name in ['small','stable','divergent']:
    part=[r for r in rows if r['experiment']==name]
    plt.semilogy([r['epoch'] for r in part],[r['mse']+1e-16 for r in part],label=name)
plt.title('Learning-rate comparison')
plt.xlabel('Epoch');plt.ylabel('MSE (log scale)')
plt.legend();plt.grid(alpha=.2)
plt.tight_layout();plt.savefig(output/'learning_rate_comparison.png',dpi=160)
plt.show()
print('저장 완료:',output.resolve())
```

## 4. 결과 해석 문장 구성법

1. **실험조건**: 어떤 데이터/모델/초기값/학습률을 사용했는가?
2. **관찰**: 그래프나 출력 숫자에 무엇이 보이는가?
3. **원인**: 어떤 수식/원리로 이를 설명할 수 있는가?
4. **한계**: 이 현상을 다른 데이터에도 일반화할 수 있는가?

예: “`lr=.25`는 이 예제에서 이론적 상한 `3/14`보다 커서 손실이 증가했다. 다만 이는 **특정 이차 손실 함수의 조건**이고, 복잡한 신경망 전체에 같은 한계를 적용할 수는 없다.”

## 5. 재현성 체크

- `np.random.default_rng(42)`로 난수 시드를 고정했다.
- `pip install` 또는 `requirements.txt`를 제공한다.
- 생성된 파일 경로를 README에 쓴다.
- 축 단위와 로그 축 여부를 표시한다.
- `assert` 기준과 출력된 오차를 함께 제공한다.

---

## 직접 풀어보기

① `training_history.csv`를 열어 실험별 마지막 행을 비교하라. ② 그래프에 기준선을 하나 추가하라. ③ “결과는 좋았다” 대신 관측값과 이론 근거를 포함한 두 문장을 써 보라.

<details>
<summary>정답·힌트 확인</summary>

① 실험별 에포크 34의 `mse`를 비교한다. ② 예: `plt.axhline(1e-3,ls="--")`. ③ 조건·수치·원인·한계를 각각 명시하면 된다.

</details>

## 미션 평가 체크

- [ ] 평가표의 각 행에 대응하는 코드, 수치, 이미지, 설명을 모두 마련했는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 6-3 경사하강법·Momentum 실험](STEP_6-3_경사하강법·Momentum_실험.md) · [전체 목차](../README.md) · [다음: 6-5 Notebook·README 작성 및 제출 점검 →](STEP_6-5_Notebook·README_작성_및_제출_점검.md)
