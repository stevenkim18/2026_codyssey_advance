# STEP 6-5. Notebook·README 작성 및 제출 점검

> 코디세이 A2-1 AI 수학 · STEP 6 통합 실습과 제출 · Python/NumPy 입문자용

## 학습 목표

- GitHub 저장소 폴더 구조와 실행 방법을 정리한다.
- Notebook과 README가 맡는 역할을 구분한다.
- 평가 체크리스트로 최종 제출을 점검한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 평가자가 첫 3분 안에 이해할 수 있게 만들자

프로젝트가 좋아도 **실행할 수 없으면 평가가 어렵다**. README에 이 프로젝트의 목표, 환경 설치법, 코드 실행 순서, 생성 결과의 위치를 적는다.

권장 구조:

```text
A2-1-ai-math/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_linear_algebra.ipynb
│   ├── 02_calculus_backprop.ipynb
│   └── 03_optimization.ipynb
├── src/
│   ├── linear_algebra.py
│   ├── calculus_backprop.py
│   └── optimization.py
├── outputs/
│   ├── svd_comparison.png
│   └── optimization_comparison.png
└── docs/
    └── step1/ ... step6/
```

모든 파일을 반드시 만들 필요는 없지만, 실제로 존재하지 않는 파일을 README에서 실행하라고 안내해서는 안 된다.

## 2. 최소 requirements.txt

```text
numpy
matplotlib
```

실행 명령(최종 파일명에 맞게 변경):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/linear_algebra.py
python src/calculus_backprop.py
python src/optimization.py
```

Notebook 사용 시 `jupyterlab`을 추가하고 `jupyter lab`으로 실행할 수 있다. `.ipynb`는 마크다운 설명, 코드, 실행 결과를 하나의 파일로 저장하는 데 유용하다.

## 3. README 복사 템플릿

```markdown
# A2-1 AI가 어떻게 학습하는지 수학으로 직접 풀어보기

## 목표
NumPy로 선형대수/미적분/역전파/최적화/확률을 구현하고 검증합니다.

## 환경
Python 3.10 이상 권장, NumPy, Matplotlib

## 실행
1. `python -m pip install -r requirements.txt`
2. `python src/linear_algebra.py`
3. `python src/calculus_backprop.py`
4. `python src/optimization.py`

## 핵심 검증
| 항목 | 결과 | 기준 |
|---|---|---|
| det 면적비 상대오차 | (출력값 기입) | 1% 이내 |
| Power Iteration 상대오차 | (출력값 기입) | 5% 이내 |
| 수치미분 절대오차 | (출력값 기입) | 1e-4 이내 |

## 결과 그래프
실행 시 생성하는 PNG 파일을 `outputs/`에 저장합니다.

## 한계
데모용 작은 데이터와 단순한 신경망을 사용한 교육용 실험입니다.
```

## 4. 실행 코드에서 Notebook 만들기: 기본 원리

Notebook은 `.ipynb` 확장자를 사용하며, 셀 배열을 포함하는 JSON 형식이다. Jupyter 환경이 있다면 이미 실행하던 `.py`를 셀 단위로 나누고, 각 셀 앞에 설명을 넣어도 된다.

```python
# Jupyter에서 바로 확인할 수 있는 가장 작은 형태의 코드 셀
import numpy as np
x = np.array([1., 2., 3.])
print('평균:', np.mean(x))
```

수식의 유도와 설명은 Markdown 셀에 적는다. 실행 결과의 이미지가 빠지지 않게 모든 셀을 위에서부터 다시 실행하고 저장한다.

## 5. 최종 제출 체크리스트

- [ ] 회전·스케일링·전단 그래프 3종 출력
- [ ] 행렬식 면적비 상대오차 1% 이내
- [ ] Power Iteration 상대오차 5% 이내
- [ ] SVD `k=10/50/100` 한 Figure
- [ ] 수치미분 절대오차 1e-4 이내 출력
- [ ] Gradient 화살표가 등고선에 수직임을 확인
- [ ] 2→2→1 Sigmoid+BCE 순전파/역전파 값 및 검증
- [ ] GD·Momentum, 학습률 수렴/발산 비교
- [ ] 각 그래프의 제목·축·범례 작성
- [ ] README 실행 명령 실제 확인
- [ ] Notebook 모든 셀 처음부터 실행 가능
- [ ] 결과 이미지를 포함해 GitHub에 업로드

## 6. 최종 발표 구성(약 5분)

1. **미션 목표(30초)**: 수학을 구현과 검증으로 이해했다.
2. **선형대수(1분)**: 변환·행렬식·Power Iteration·SVD.
3. **미적분과 역전파(1분 30초)**: 수치미분과 2→2→1 기울기.
4. **최적화(1분)**: 학습률, GD, Momentum.
5. **배운 점(1분)**: 수식→코드→검증의 연결, 데이터/설정의 한계.

---

## 직접 풀어보기

① 실제 저장소 구조에 맞게 README 실행 명령을 수정하라. ② 모든 코드/Notebook 셀을 처음부터 끝까지 실행하라. ③ 평가표의 수치를 복사하지 말고 실행 결과에서 새로 기록하라.

<details>
<summary>정답·힌트 확인</summary>

① 파일과 명령이 일치해야 한다. ② 첫 실행에서 누락된 의존성이나 변수 참조를 점검한다. ③ 평가자는 재현 가능한 실험을 확인하려는 것이다.

</details>

## 미션 평가 체크

- [ ] README, 코드, Notebook, 출력 이미지, 핵심 검증 수치를 실제 파일과 일치하도록 점검했는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 6-4 결과 그래프·해석 정리](STEP_6-4_결과_그래프·해석_정리.md) · [전체 목차](../README.md)
