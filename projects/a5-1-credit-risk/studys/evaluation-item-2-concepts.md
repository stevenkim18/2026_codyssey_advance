# A5-1 항목 2: 코드 구조와 데이터 흐름 설명하기

이 문서는 [평가 원문](../../../evaluations/evaluations_a5-1.md)의 **항목 2 평가 문장 세 개**를 설명하기 위한 자료입니다. [항목 1 개념 설명](evaluation-item-1-concepts.md)과 같은 방식으로 평가 의도, 쉬운 개념, 실제 코드, 발표 예시를 연결했습니다.

항목 2의 핵심은 **“코드를 실행했다”에서 한 걸음 더 나아가, 각 부분의 역할과 데이터가 이동하는 순서를 코드로 설명하는 것**입니다. 코드 조각은 해당 함수의 핵심 부분을 발췌하거나 설명을 위해 축약했습니다. 독립 실행용 전체 코드는 아닙니다.

## 먼저 알아둘 용어

| 용어 | 쉬운 설명 | 예 |
|---|---|---|
| 모듈 | 관련된 기능을 모아 놓은 Python 파일 | `preprocess.py` |
| 함수 | 입력을 받아 정해진 일을 수행하는 코드 묶음 | `split_data(df)` |
| 전처리 | 모델에 넣기 전에 입력 형태와 크기를 정리 | 결측치 대체, 표준화, 인코딩 |
| 학습 | 데이터에서 예측 기준을 찾는 과정 | `model.fit(X_train, y_train)` |
| 추론·예측 | 학습한 기준으로 새 입력에 답하는 과정 | `model.predict(X_test)` |
| 평가 | 예측과 정답을 비교해 성능을 계산 | F1, AUC, RMSE |
| `X`와 `y` | 입력 특성 표와 정답 | 고객 정보와 연체 여부 |

비유하면 데이터는 재료, 전처리는 손질, 학습은 조리법을 배우는 과정, 예측은 그 조리법으로 새 재료를 처리하는 과정입니다. 평가는 완성된 결과가 정답과 얼마나 가까운지 확인하는 일입니다.

## 1. 전처리 → 학습 → 평가를 어떻게 분리했는가?

> **평가 문장:** 전처리 → 학습 → 평가 흐름을 어떤 구조로 분리했는지 설명해 주세요. 각 모듈/셀/함수의 역할과 흐름을 코드를 보며 설명할 수 있는가?

### 평가에서 확인하는 것

파일 이름을 외우는 것이 아니라 **각 코드가 무엇을 받아 무엇을 만들고, 다음 단계가 그것을 어떻게 사용하는지** 설명할 수 있는지 확인합니다. 이 프로젝트는 노트북 셀 대신 Python 모듈과 함수로 역할을 나눴습니다.

### 역할별 코드 지도

| 모듈 | 주요 함수 | 입력 → 결과 | 역할 |
|---|---|---|---|
| [data_gen.py](../src/data_gen.py) | `generate_data()`, `main()` | 생성 규칙 → DataFrame·CSV | 가상 고객 데이터 생성 |
| [preprocess.py](../src/preprocess.py) | `load_data()` | CSV → DataFrame | 데이터 읽기 |
| 같은 모듈 | `add_age_group()`, `split_data()` | DataFrame → 분할 묶음 `Split` | 연령대 생성과 Train/Test 분할 |
| 같은 모듈 | `build_preprocessor()` | 전처리 설정 → 학습 전 변환기 | 공통 전처리 구성 |
| [baseline.py](../src/baseline.py) | `run_baseline()` | `Split` → 규칙 예측·지표 | 사람이 만든 규칙으로 비교 기준 생성 |
| [classification.py](../src/classification.py) | `make_logreg()`, `make_forest()` | 모델 설정 → 학습 전 Pipeline | 분류 모델 구성 |
| 같은 모듈 | `tune_forest()` | Train → 최적 설정과 학습된 모델 | 분류 모델 튜닝 |
| 같은 모듈 | `evaluate()` | 학습 모델·평가 입력·정답 → 예측·지표 | 분류 평가 |
| [regression.py](../src/regression.py) | `make_model()`, `run_regression()` | `Split` → 회귀 모델·계수·지표 | 회귀 학습·검증·평가 |
| [metrics.py](../src/metrics.py) | `classification_metrics()`, `regression_metrics()` | 실제값·예측값 → 지표 | 지표 계산 공통화 |
| [plotting.py](../src/plotting.py) | `apply_style()`, `save_figure()` | 그림 → 공통 스타일·PNG | 시각화 표현·저장 공통화 |

`classification.py`와 `regression.py`는 학습만 하는 파일은 아닙니다. 각 과제의 학습·평가 흐름을 관리하는 역할도 합니다. 그래프의 내용은 각 모델 모듈에서 만들고, 공통 스타일과 저장은 `plotting.py`가 담당합니다.

### 실제 데이터 흐름

```text
data_gen.main()
    └─ generate_data() → data/finance_data.csv
                                  ↓
                             load_data()
                                  ↓
                             split_data()
                                  ↓
                           Split: 6개 묶음
                  ┌───────────────┴────────────────┐
                  ↓                                ↓
        X_train + y_clf_train            X_train + y_reg_train
                  ↓                                ↓
       분류 Pipeline 학습·튜닝             회귀 Pipeline 학습·Alpha 선택
                  ↓                                ↓
         X_test로 연체 예측                 X_test로 신용 점수 예측
                  ↓                                ↓
         y_clf_test와 비교                   y_reg_test와 비교
                  ↓                                ↓
        F1·AUC·혼동 행렬                    RMSE·MAE·R²·계수 그래프
```

`Split`은 다음 여섯 값을 이름으로 꺼낼 수 있게 묶은 구조입니다.

| 값 | 내용 |
|---|---|
| `X_train`, `X_test` | 학습·평가용 입력 |
| `y_clf_train`, `y_clf_test` | 학습·평가용 연체 정답 |
| `y_reg_train`, `y_reg_test` | 학습·평가용 신용 점수 정답 |

입력과 두 정답을 한 번에 분할하므로 같은 고객의 정보와 정답이 서로 어긋나지 않습니다.

### 코드를 보며 설명하는 순서

1. `preprocess.py`의 `split_data()`에서 입력과 두 타깃을 어떻게 나누는지 보여줍니다.
2. `build_preprocessor()`에서 수치형·범주형 처리 경로를 보여줍니다.
3. `classification.py`의 `make_logreg()` 또는 `make_forest()`에서 전처리와 모델 연결을 보여줍니다.
4. `main()`의 `.fit(split.X_train, split.y_clf_train)`에서 학습 입력을 짚습니다.
5. `evaluate()`에서 Test 예측과 지표 계산으로 이어지는 흐름을 보여줍니다.
6. 회귀 쪽은 `run_regression()`에서 `y_reg_train`·`y_reg_test`를 사용하는 차이를 설명합니다.

분류의 핵심 흐름을 축약하면 다음과 같습니다.

```python
split = split_data(load_data())
model = make_logreg("balanced")  # 전처리 + 분류 모델 구성
model.fit(split.X_train, split.y_clf_train)
result = evaluate(model, split.X_test, split.y_clf_test)
```

`make_logreg()`는 학습 전 작업 순서를 만들고, `fit()`이 실제로 학습합니다. **구성과 학습은 서로 다른 단계**입니다.

### 이렇게 분리한 이유

- 전처리를 수정할 때 회귀·분류 파일마다 같은 코드를 고칠 필요가 줄어듭니다.
- 모델을 바꾸면서 같은 분할과 지표를 유지하기 쉽습니다.
- 잘못된 결과가 나오면 데이터·전처리·학습·평가 중 어느 단계인지 추적하기 쉽습니다.
- 지표 계산을 공통화해 모델마다 다른 계산식을 쓰는 실수를 줄입니다.

### 이렇게 설명해 보세요

> 데이터 읽기와 분할, 공통 전처리는 preprocess.py에 모았습니다. 분류와 회귀 모듈은 각각 모델을 구성하고 학습·평가를 진행하며, 지표 계산은 metrics.py를 공통으로 사용합니다. 따라서 같은 고객 분할을 유지하면서 모델만 바꾸어 비교할 수 있습니다.

## 2. 전처리의 `fit`은 어디서 수행하고 누수는 어떻게 막았는가?

> **평가 문장:** 데이터 누수(Data Leakage) 방지를 위해 전처리 기준(`fit`)을 어디서 수행했는지, 해당 코드를 직접 가리키며 설명할 수 있는가?

### 평가에서 확인하는 것

“Train에만 fit했습니다”라는 말과 실제 코드가 일치하는지 확인합니다. 특히 **최종 Train/Test 분할뿐 아니라 교차검증의 각 폴드 안에서도 전처리 기준이 분리되는지** 설명해야 합니다.

### `fit`이 학습하는 기준

| 전처리 | `fit`에서 결정 | `transform`에서 수행 |
|---|---|---|
| 중앙값 대체 | 학습 열의 중앙값 | 결측치를 그 중앙값으로 채움 |
| 표준화 | 학습 열의 평균·표준편차 | 같은 평균·표준편차로 변환 |
| 원-핫 인코딩 | 학습 데이터의 범주 목록 | 학습한 범주에 대응하는 열 생성 |

**가상 예시:** Train 소득이 3,000·5,000이면 평균은 4,000입니다. Test 소득 10,000까지 넣어 평균 6,000을 계산하면 시험 데이터의 정보가 학습 기준에 들어갑니다. 이것이 전처리 누수입니다.

Test는 별도로 자신의 평균을 계산해 변환하는 것도 아닙니다. **Train에서 얻은 하나의 기준을 Test에도 적용**해야 합니다.

### 근거 1: `build_preprocessor()`는 구성만 한다

[preprocess.py](../src/preprocess.py)의 `build_preprocessor()`는 `SimpleImputer`, `StandardScaler`, `OneHotEncoder`를 `ColumnTransformer`로 묶어 반환합니다. 함수 안에서 입력 데이터에 대한 `fit()`은 수행하지 않습니다.

```python
numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
```

이 코드는 “어떤 처리를 할 것인가”를 정합니다. 중앙값·평균을 실제로 계산하는 시점은 이후 Pipeline의 `fit()`입니다.

### 근거 2: 전처리와 모델을 함께 Train으로 학습한다

[classification.py](../src/classification.py)의 `make_forest()`는 다음 구조를 반환합니다.

```python
Pipeline([("pre", build_preprocessor()), ("rf", forest)])
```

`main()`에서 기본 랜덤 포레스트를 학습하는 코드는 다음과 같습니다.

```python
forest_default = make_forest(n_estimators=100, n_jobs=-1).fit(
    split.X_train, split.y_clf_train
)
```

이 `fit()`에서 앞 단계 전처리는 `X_train`으로 기준을 학습하고 데이터를 변환합니다. 마지막 모델은 변환된 입력과 `y_clf_train`으로 학습합니다.

평가 함수 `evaluate()`는 `predict_proba(X)`로 위험 점수를 구합니다. Test를 넣으면 전처리는 이미 학습한 기준으로 변환하고 모델이 예측합니다. Test로 전처리를 다시 `fit`하지 않습니다.

### 근거 3: 교차검증 안에서도 전처리를 다시 학습한다

`tune_forest()`의 핵심은 전처리를 포함한 **전체 Pipeline**을 `GridSearchCV`에 전달하는 것입니다.

```python
search = GridSearchCV(
    make_forest(),
    PARAM_GRID,
    scoring=SCORING,
    refit="f1",
    cv=make_cv(),
    n_jobs=-1,
)
return search.fit(split.X_train, split.y_clf_train)
```

```text
Train 내부의 한 교차검증 반복
  학습 폴드 → 전처리 fit·변환 → 모델 fit
  검증 폴드 → 학습 폴드의 전처리로 변환 → 예측·점수 계산
```

Train 전체를 먼저 표준화한 뒤 교차검증하면, 최종 Test를 보지 않았어도 검증 폴드의 정보가 전처리에 들어갑니다. Pipeline을 검증 안에 넣는 이유입니다.

[regression.py](../src/regression.py)의 `run_regression()`도 `cross_validate()`에 `make_model(kind, alpha)`로 만든 전체 Pipeline을 전달합니다. Alpha 선택은 CV RMSE로 하고 Test 지표는 보고용입니다. 여러 Alpha의 Test 결과를 기록하더라도 그것을 보고 Alpha를 고르면 안 됩니다.

### 함께 짚을 세 가지 예외와 주의점

| 코드·상황 | 설명 |
|---|---|
| `preprocess.main()`의 `build_preprocessor().fit(split.X_train)` | 독립 실행 시 전처리 확인용이며, 이 경우에도 Train으로만 학습 |
| `add_age_group()`을 분할 전에 실행 | 20·30·40 같은 경계가 고정되어 데이터에서 기준을 학습하지 않으므로 이 변환 자체는 전처리 누수가 아님 |
| 입력에서 두 타깃 제외 | `FEATURES`에는 `credit_score`와 `is_overdue`가 없어 정답을 입력으로 주는 실수 방지 |

이 데이터의 연체 정답은 신용 점수를 이용해 생성됩니다. 따라서 신용 점수를 분류 입력에 넣으면 문제 정의가 달라지고 훨씬 유리한 정보를 쓰게 됩니다. 다만 현실에서 신용 점수가 실제 예측 시점에 사용 가능한 정보라면 사용 자체가 항상 누수인 것은 아닙니다. **정보의 생성 시점과 과제의 입력 정의**를 함께 확인해야 합니다.

SMOTE도 전체 분할 전이 아니라 `imblearn` Pipeline의 학습 단계에서만 실행합니다. Test 예측 시에는 샘플을 추가하지 않습니다.

### 이렇게 설명해 보세요

> build_preprocessor는 전처리 순서만 만들고 데이터를 학습하지 않습니다. 실제 기준은 모델 Pipeline에 Train을 넣어 fit할 때 계산합니다. 튜닝에도 전체 Pipeline을 전달해서 각 학습 폴드에서 기준을 다시 계산합니다. Test에는 학습한 기준을 적용해 예측만 수행합니다.

## 3. 공통 전처리와 회귀·분류 모델을 어떻게 함께 설계했는가?

> **평가 문장:** 회귀 모델과 분류 모델을 동시에 다루는 구조에서, 공통 전처리와 개별 모델링을 어떻게 설계했는지 설명할 수 있는가?

### 평가에서 확인하는 것

재사용할 부분과 문제에 따라 달라질 부분을 구분했는지 확인합니다. 같은 고객 데이터로 두 문제를 풀더라도 **정답·모델·평가 지표는 다릅니다.**

### 공통으로 쓰는 부분과 따로 만드는 부분

| 구분 | 회귀와 분류에서의 처리 |
|---|---|
| 데이터 로딩 | 같은 CSV를 읽음 |
| 고객 분할 | 같은 `split_data()`로 같은 고객을 Train/Test에 배치 |
| 입력 특성 | 같은 `FEATURES` 사용, 두 타깃 모두 제외 |
| 전처리 구성 | 같은 `build_preprocessor()` 재사용 |
| 학습된 전처리 객체 | 각 Pipeline 안에서 별도로 생성·학습 |
| 정답 | 회귀는 신용 점수, 분류는 연체 여부 |
| 모델 | 회귀는 Ridge/Lasso, 분류는 로지스틱 회귀/랜덤 포레스트 |
| 설정 선택 | 회귀는 CV RMSE, 랜덤 포레스트는 CV F1 |
| 평가 | 회귀는 RMSE·MAE·R², 분류는 Accuracy·F1·AUC 등 |

**공통 전처리를 쓴다는 말은 학습된 객체 하나를 두 모델이 공유한다는 뜻이 아닙니다.** 같은 설계 함수를 호출해 각자 새로운 객체를 만듭니다. 한 모델을 다시 학습해도 다른 모델의 전처리 상태가 덮어써지지 않도록 합니다.

### 코드 연결하기

회귀의 `make_model()`은 다음 구조를 만듭니다.

```python
Pipeline([("pre", build_preprocessor()), ("model", reg)])
```

분류의 `make_forest()`도 같은 전처리 함수를 호출하지만 마지막 모델이 다릅니다.

```python
Pipeline([("pre", build_preprocessor()), ("rf", forest)])
```

```text
공통 설계 함수 build_preprocessor()
      ├─ 새 전처리 객체 A + Ridge → y_reg_train으로 학습
      ├─ 새 전처리 객체 B + Lasso → y_reg_train으로 학습
      ├─ 새 전처리 객체 C + LogReg → y_clf_train으로 학습
      └─ 새 전처리 객체 D + Forest → y_clf_train으로 학습
```

### 분할과 교차검증에서 알아둘 점

`split_data()`는 연체 비율을 유지하도록 `stratify=df[TARGET_CLF]`를 사용합니다. 이를 **층화 분할**이라고 합니다. 회귀 정답도 같은 호출에 넣어 동일한 고객 행을 유지합니다.

회귀의 교차검증은 다음처럼 연체 여부로 나눈 폴드 인덱스를 사용합니다.

```python
cv_splits = list(make_cv().split(split.X_train, split.y_clf_train))
```

이때 회귀가 연체를 정답으로 학습하는 것은 아닙니다. **연체 여부는 폴드를 구성하는 데 사용하고, 실제 회귀 학습 정답은 신용 점수**입니다. 연속형 신용 점수를 그대로 `StratifiedKFold`의 클래스 정답으로 넣은 것도 아닙니다.

회귀·분류는 서로 다른 문제이므로 RMSE 30과 F1 0.64 중 어느 것이 더 좋은 모델인지 직접 비교할 수 없습니다. 같은 문제·같은 지표 안에서 비교해야 합니다.

### 이렇게 설명해 보세요

> 데이터 로딩, 고객 분할, 전처리 설계는 공통으로 사용했습니다. 다만 학습된 객체를 공유하는 대신 각 모델에 새 전처리 객체를 넣었습니다. 회귀는 신용 점수를 정답으로 학습해 RMSE로 설정을 고르고, 분류는 연체 여부를 학습해 F1과 AUC로 평가합니다. 두 문제 모두 같은 고객을 평가해 입력 조건을 맞췄습니다.

## 발표 전 확인할 질문

1. CSV에서 읽은 데이터가 어떤 함수를 거쳐 결과표까지 가는지 보여줄 수 있는가?
2. 전처리 객체를 만드는 코드와 실제 `fit()`하는 코드를 구분하는가?
3. Test뿐 아니라 교차검증의 검증 폴드에도 누수가 없는 이유를 설명하는가?
4. 고정 연령대 변환과 데이터로 평균을 학습하는 변환의 차이를 아는가?
5. 공통 전처리 설계와 학습된 객체 공유의 차이를 설명하는가?
6. 회귀와 분류가 같은 고객을 쓰면서 정답과 지표는 다르다는 점을 설명하는가?

관련 문서: [항목 1: 구현과 결과물](evaluation-item-1-concepts.md), [항목 3: 이론과 결과 해석](evaluation-item-3-concepts.md), [프로젝트 README](../README.md).
