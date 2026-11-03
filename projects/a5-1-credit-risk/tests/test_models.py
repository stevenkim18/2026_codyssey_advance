"""베이스라인, 회귀, 분류 모델 테스트: 과제의 필수 요구사항(규칙 5개, Alpha, 불균형 처리, 탐색 범위)을 확인한다."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.baseline import (
    RULE_NAMES,
    count_rule_hits,
    rule_based_predict,
    rule_flags,
    run_baseline,
)
from src.classification import (
    PARAM_GRID,
    count_grid_combinations,
    evaluate,
    make_forest,
    make_logreg,
)
from src.regression import ALPHAS, SCORE_MAX, SCORE_MIN, make_model, predict_score


class TestBaseline:
    def test_has_at_least_five_rules(self):
        row = pd.Series(
            {"overdue_count_6m": 0, "debt_ratio": 0.0, "annual_income": 9000, "credit_card_count": 3}
        )
        assert len(RULE_NAMES) >= 5
        assert len(rule_flags(row)) == len(RULE_NAMES) >= 5

    def test_rule_counting(self):
        safe = pd.Series({"overdue_count_6m": 0, "debt_ratio": 0.1, "annual_income": 8000})
        risky = pd.Series({"overdue_count_6m": 3, "debt_ratio": 0.9, "annual_income": 2000})
        assert count_rule_hits(safe) == 0
        assert count_rule_hits(risky) == 5  # 다섯 규칙이 모두 걸린다
        assert rule_based_predict(safe, min_hits=2) == 0
        assert rule_based_predict(risky, min_hits=2) == 1

    def test_metrics_are_reported(self, split):
        result = run_baseline(split)
        assert set(result["metrics"]) == {"Accuracy", "Precision", "Recall", "F1-Score", "AUC"}
        assert 1 <= result["min_hits"] <= 5
        assert len(result["y_pred"]) == len(split.X_test)
        assert result["metrics"]["F1-Score"] > 0.3  # 아무 규칙도 못 맞히는 수준은 아니다


class TestRegression:
    def test_alpha_grid_matches_mission(self):
        assert ALPHAS == [0.01, 0.1, 1, 10, 100]

    def test_prediction_is_clipped_to_score_range(self, split):
        model = make_model("Ridge", 1).fit(split.X_train, split.y_reg_train)
        extreme = split.X_test.head(3).copy()
        extreme["annual_income"] = 1e9  # 점수가 1000 을 훨씬 넘을 입력
        raw = model.predict(extreme)
        assert raw.max() > SCORE_MAX
        pred = predict_score(model, extreme)
        assert pred.min() >= SCORE_MIN and pred.max() <= SCORE_MAX

    def test_lasso_zeroes_coefficients_but_ridge_does_not(self, split):
        def n_zero(kind, alpha):
            model = make_model(kind, alpha).fit(split.X_train, split.y_reg_train)
            return int((np.abs(model[-1].coef_) < 1e-8).sum())

        assert n_zero("Lasso", 10) > n_zero("Lasso", 0.01)  # 규제가 세질수록 0 이 늘어난다
        assert n_zero("Lasso", 100) == 10  # 너무 세면 전부 0 (모델이 상수만 낸다)
        assert n_zero("Ridge", 100) == 0  # Ridge 는 0 에 가깝게 줄일 뿐 0 으로 만들지 않는다

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            make_model("ElasticNet", 1)


class TestClassification:
    def test_grid_search_space_within_guide(self):
        assert count_grid_combinations(PARAM_GRID) <= 100
        assert all(len(values) <= 5 for values in PARAM_GRID.values())

    def test_imbalance_handling_is_applied(self):
        assert make_logreg("balanced")[-1].class_weight == "balanced"
        assert make_logreg("none")[-1].class_weight is None
        assert make_forest()[-1].class_weight == "balanced"

    def test_smote_lives_inside_the_pipeline(self, split):
        model = make_logreg("smote")
        # imblearn Pipeline 안에 있어서 fit 때 Train 에서만 샘플을 만들고, predict 때는 건너뛴다
        assert isinstance(model, ImbPipeline)
        assert isinstance(model.named_steps["smote"], SMOTE)
        model.fit(split.X_train, split.y_clf_train)
        assert model.predict_proba(split.X_test).shape == (len(split.X_test), 2)

    def test_unknown_handling_is_rejected(self):
        with pytest.raises(ValueError):
            make_logreg("oversample")

    def test_forest_probabilities_and_metrics(self, split):
        model = make_forest(n_estimators=30, max_depth=6, n_jobs=-1)
        model.fit(split.X_train, split.y_clf_train)
        out = evaluate(model, split.X_test, split.y_clf_test)
        assert out["y_score"].min() >= 0 and out["y_score"].max() <= 1
        assert set(np.unique(out["y_pred"])) <= {0, 1}
        assert out["metrics"]["AUC"] > 0.85
