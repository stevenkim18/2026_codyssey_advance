"""편향·분산 분석과 속도 측정 테스트. 속도는 실행마다 달라지므로 크기 비교 대신 구조만 확인한다."""

import numpy as np
from src.classification import PARAM_GRID, make_forest, make_logreg
from src.tradeoff import (
    TUNED_FOREST_PARAMS,
    bootstrap_indices,
    cv_gap_table,
    make_tree,
    measure_serving,
    prediction_wobble,
)


def test_tuned_params_are_inside_the_search_space():
    # 분석에 쓰는 랜덤 포레스트 설정이 GridSearchCV 후보에 실제로 있는 값인지 확인한다
    for name, value in TUNED_FOREST_PARAMS.items():
        assert value in PARAM_GRID[f"rf__{name}"]


def test_bootstrap_indices_are_reproducible_and_shared():
    first = bootstrap_indices(100, 3)
    second = bootstrap_indices(100, 3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(len(idx) == 100 and idx.max() < 100 for idx in first)  # 복원추출이라 길이는 같다
    assert len(np.unique(first[0])) < 100  # 복원추출이면 중복이 생긴다


def test_forest_wobbles_less_than_a_single_tree_of_the_same_depth(split):
    tree = prediction_wobble(make_tree(max_depth=12), split, n_bootstrap=4)
    forest = prediction_wobble(make_forest(n_estimators=40, max_depth=12, n_jobs=1), split, n_bootstrap=4)
    assert forest["예측 확률 흔들림"] < tree["예측 확률 흔들림"]  # 평균을 내면 분산이 줄어든다
    assert forest["재학습 Test AUC 평균"] > tree["재학습 Test AUC 평균"]


def test_unlimited_tree_overfits_more_than_logreg(split):
    table = cv_gap_table({"tree": make_tree(), "logreg": make_logreg("balanced")}, split)
    assert table.loc["tree", "Train F1"] > 0.99  # 제한 없는 트리는 Train 을 외운다
    assert table.loc["tree", "격차"] > 0.3  # 그래서 CV 점수와 크게 벌어진다
    assert table.loc["logreg", "격차"] < 0.05  # 단순한 모델은 격차가 거의 없다


def test_measure_serving_reports_consistent_latencies(split):
    model = make_logreg("balanced").fit(split.X_train, split.y_clf_train)
    out = measure_serving(
        lambda X: model.predict_proba(X),
        lambda X, i: model.predict_proba(X.iloc[[i]]),
        split.X_test,
        n_batch=2,
        n_single=10,
    )
    assert out["1건 p50 (ms)"] > 0
    assert out["1건 p95 (ms)"] >= out["1건 p50 (ms)"]
    assert out["코어당 요청 (건/초)"] == int(1000 / out["1건 p50 (ms)"])
    assert out["배치 처리량 (건/초)"] > 0
