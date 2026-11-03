"""앙상블의 편향·분산과 운영 선택 분석 (A5-1).

두 가지 질문에 숫자로 답한다.
    1. 앙상블은 왜 단일 트리보다 나은가?
       같은 깊이의 트리 한 그루와 랜덤 포레스트를 비교해 분산 감소를 보인다.
       - Train 점수와 교차 검증 점수의 격차: 클수록 학습 데이터에 과하게 맞춘(분산이 큰) 모델이다.
       - 학습 데이터를 바꿔 20번 다시 학습했을 때 같은 Test 고객의 예측 확률이 얼마나 흔들리는가.
         모델마다 같은 20개의 재표본(부트스트랩)을 쓴다.
    2. 서비스에서는 어떤 모델을 쓸 것인가?
       모델별 응답 시간, 처리량, 모델 크기, 학습 시간을 잰다.

속도는 이 컴퓨터에서 스레드 1개로 잰 값이다. 절대값은 환경마다 다르고, 모델 사이의 비율을 본다.
랜덤 포레스트의 튜닝 결과(`TUNED_FOREST_PARAMS`)는 `src.classification` 의 GridSearchCV 가 고른 값이다.

실행: `uv run python -m src.tradeoff` (프로젝트 폴더에서, 약 1분)
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.parallel import Parallel, delayed

from src.baseline import rule_based_predict, run_baseline
from src.classification import make_forest, make_logreg
from src.metrics import to_markdown_table
from src.preprocess import (
    RANDOM_STATE,
    Split,
    build_preprocessor,
    load_data,
    make_cv,
    split_data,
)

TUNED_FOREST_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 1}
N_BOOTSTRAP = 20  # 학습 데이터를 바꿔 다시 학습하는 횟수


def make_tree(**params) -> Pipeline:
    """전처리 → 결정 트리 한 그루 (랜덤 포레스트와 같은 class_weight='balanced')."""
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE, **params)
    return Pipeline([("pre", build_preprocessor()), ("tree", tree)])


def bias_variance_models() -> dict[str, Pipeline]:
    """편향·분산 비교 대상. 트리는 얕은 것(편향 큼)에서 제한 없는 것(분산 큼)까지, 이어서 랜덤 포레스트."""
    return {
        "트리 depth=3 (얕음)": make_tree(max_depth=3),
        "트리 depth=12": make_tree(max_depth=12),
        "트리 제한 없음": make_tree(),
        "랜덤 포레스트 (튜닝: 300그루, depth=12)": make_forest(n_jobs=1, **TUNED_FOREST_PARAMS),
        "LogReg (balanced)": make_logreg("balanced"),
    }


def cv_gap_table(models: dict[str, Pipeline], split: Split) -> pd.DataFrame:
    """Train 점수와 교차 검증 점수의 격차. 같은 5 fold 를 모든 모델에 쓴다."""
    folds = list(make_cv().split(split.X_train, split.y_clf_train))
    rows = {}
    for name, model in models.items():
        cv = cross_validate(
            model,
            split.X_train,
            split.y_clf_train,
            cv=folds,
            scoring={"f1": "f1", "auc": "roc_auc"},
            return_train_score=True,
            n_jobs=-1,
        )
        rows[name] = {
            "Train F1": cv["train_f1"].mean(),
            "CV F1": cv["test_f1"].mean(),
            "격차": cv["train_f1"].mean() - cv["test_f1"].mean(),
            "CV F1 표준편차": cv["test_f1"].std(),
            "Train AUC": cv["train_auc"].mean(),
            "CV AUC": cv["test_auc"].mean(),
        }
    return pd.DataFrame(rows).T.rename_axis("모델")


def bootstrap_indices(n_rows: int, n_bootstrap: int = N_BOOTSTRAP) -> list[np.ndarray]:
    """복원추출한 행 번호 묶음. 같은 시드라서 모든 모델이 같은 재표본으로 비교된다."""
    rng = np.random.RandomState(RANDOM_STATE)
    return [rng.randint(0, n_rows, n_rows) for _ in range(n_bootstrap)]


def _fit_predict(model, X, y, idx, X_eval) -> np.ndarray:
    fitted = clone(model).fit(X.iloc[idx], y.iloc[idx])
    return fitted.predict_proba(X_eval)[:, 1]


def prediction_wobble(model, split: Split, n_bootstrap: int = N_BOOTSTRAP) -> dict[str, float]:
    """학습 데이터를 바꿔 다시 학습할 때 Test 고객별 예측 확률의 표준편차(평균). 클수록 분산이 크다."""
    indices = bootstrap_indices(len(split.X_train), n_bootstrap)
    probs = np.array(
        Parallel(n_jobs=-1)(
            delayed(_fit_predict)(model, split.X_train, split.y_clf_train, idx, split.X_test)
            for idx in indices
        )
    )
    return {
        "예측 확률 흔들림": float(probs.std(axis=0).mean()),
        "재학습 Test AUC 평균": float(np.mean([roc_auc_score(split.y_clf_test, p) for p in probs])),
    }


def bias_variance_table(models: dict[str, Pipeline], split: Split) -> pd.DataFrame:
    wobble = pd.DataFrame({name: prediction_wobble(m, split) for name, m in models.items()}).T
    return cv_gap_table(models, split).join(wobble)


def measure_serving(predict_batch, predict_one, X: pd.DataFrame, n_batch: int = 10, n_single: int = 150) -> dict:
    """배치 예측 시간과 1건 예측 시간(중앙값, 95번째 백분위)을 잰다."""
    predict_batch(X)  # 첫 호출의 준비 비용은 제외한다
    batch_ms = []
    for _ in range(n_batch):
        start = time.perf_counter()
        predict_batch(X)
        batch_ms.append((time.perf_counter() - start) * 1000)
    single_ms = []
    for i in range(n_single):
        start = time.perf_counter()
        predict_one(X, i)
        single_ms.append((time.perf_counter() - start) * 1000)
    batch = float(np.median(batch_ms))
    p50 = float(np.percentile(single_ms, 50))
    return {
        "1건 p50 (ms)": p50,
        "1건 p95 (ms)": float(np.percentile(single_ms, 95)),
        "코어당 요청 (건/초)": int(1000 / p50),  # 1건 응답 시간의 역수로 추정한 값
        f"배치 {len(X):,}건 (ms)": batch,
        "배치 처리량 (건/초)": int(len(X) / batch * 1000),
    }


def serving_table(split: Split) -> pd.DataFrame:
    """규칙 기반, 로지스틱 회귀, 랜덤 포레스트(기본/튜닝)의 속도·크기·학습 시간."""
    X = split.X_test
    rows = {}

    min_hits = run_baseline(split)["min_hits"]
    rows["규칙 기반 (if-else)"] = {
        **measure_serving(
            lambda X: X.apply(rule_based_predict, axis=1, min_hits=min_hits),
            lambda X, i: rule_based_predict(X.iloc[i], min_hits),
            X,
        ),
        "모델 크기 (KB)": "-",
        "학습 시간 (초)": "-",
    }

    candidates = {
        "LogReg (balanced)": make_logreg("balanced"),
        "랜덤 포레스트 (기본: 100그루)": make_forest(n_estimators=100, n_jobs=1),
        "랜덤 포레스트 (튜닝: 300그루)": make_forest(n_jobs=1, **TUNED_FOREST_PARAMS),
    }
    for name, model in candidates.items():
        start = time.perf_counter()
        model.fit(split.X_train, split.y_clf_train)
        fit_s = time.perf_counter() - start
        rows[name] = {
            **measure_serving(
                lambda X, m=model: m.predict_proba(X),
                lambda X, i, m=model: m.predict_proba(X.iloc[[i]]),
                X,
            ),
            "모델 크기 (KB)": int(len(pickle.dumps(model)) / 1024),
            "학습 시간 (초)": round(fit_s, 2),
        }
    # 행 단위(orient="index")로 만들어야 정수 열(건/초, KB)이 실수로 바뀌지 않는다
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("모델")


def main() -> None:
    split = split_data(load_data())

    print("[1. 편향·분산] Train 5-fold CV 와, 학습 데이터를 바꿔 20번 재학습했을 때의 흔들림")
    print(to_markdown_table(bias_variance_table(bias_variance_models(), split), floatfmt=".3f"))

    print("\n[2. 속도·크기] 이 컴퓨터, 스레드 1개, Test 2,000건")
    print(to_markdown_table(serving_table(split), floatfmt=".3f"))


if __name__ == "__main__":
    main()
