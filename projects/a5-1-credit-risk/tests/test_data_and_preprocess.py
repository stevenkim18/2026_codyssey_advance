"""데이터 생성, 분할, 전처리 파이프라인 테스트: 재현성과 데이터 누수 방지를 확인한다."""

import numpy as np
import pandas as pd
import pytest
from src.data_gen import N_SAMPLES, generate_data
from src.preprocess import (
    AGE_LABELS,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLF,
    TARGET_REG,
    add_age_group,
    build_preprocessor,
    clean_feature_names,
    load_data,
    split_data,
)


class TestDataGeneration:
    def test_shape_and_columns(self, df):
        assert df.shape == (N_SAMPLES, 8)
        assert df.isna().sum().sum() == 0

    def test_reproducible(self, df):
        pd.testing.assert_frame_equal(df, generate_data())

    def test_overdue_is_imbalanced(self, df):
        rate = df[TARGET_CLF].mean()
        assert 0.10 <= rate <= 0.15  # 과제가 말한 약 10~15% 불균형

    def test_value_ranges(self, df):
        assert df["credit_score"].between(0, 1000).all()
        assert (df["annual_income"] >= 1500).all()

    def test_overdue_is_derived_from_credit_score(self, df):
        # 연체는 신용 점수 하위 15% 안에서만 나온다 → credit_score 는 분류의 정답 힌트(누수 위험)
        threshold = df["credit_score"].quantile(0.15)
        assert (df.loc[df[TARGET_CLF] == 1, "credit_score"] < threshold).all()

    def test_load_data_missing_file_gives_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data_gen"):
            load_data(tmp_path / "none.csv")


class TestSplit:
    def test_8_to_2(self, split, df):
        assert len(split.X_train) == 8000
        assert len(split.X_test) == 2000
        assert len(split.X_train) + len(split.X_test) == len(df)

    def test_stratified(self, split):
        assert split.y_clf_train.mean() == pytest.approx(split.y_clf_test.mean(), abs=0.002)

    def test_same_rows_for_both_targets(self, split):
        assert split.X_test.index.equals(split.y_clf_test.index)
        assert split.X_test.index.equals(split.y_reg_test.index)
        assert split.X_train.index.equals(split.y_reg_train.index)

    def test_train_and_test_do_not_overlap(self, split):
        assert set(split.X_train.index).isdisjoint(split.X_test.index)

    def test_reproducible(self, df, split):
        again = split_data(df)
        assert again.X_test.index.equals(split.X_test.index)

    def test_targets_are_not_features(self, split):
        assert TARGET_REG not in FEATURES and TARGET_CLF not in FEATURES
        assert list(split.X_train.columns) == FEATURES
        assert len(FEATURES) == 6


class TestAgeGroup:
    def test_replaces_age_with_decade_labels(self, df):
        out = add_age_group(df)
        assert "age" not in out.columns
        assert set(out["age_group"]) == set(AGE_LABELS)
        assert "age" in df.columns  # 원본은 건드리지 않는다

    def test_boundaries(self):
        ages = pd.DataFrame({"age": [20, 29, 30, 69]})
        assert add_age_group(ages)["age_group"].tolist() == ["20s", "20s", "30s", "60s"]


class TestPreprocessor:
    def test_fit_on_train_only(self, split):
        pre = build_preprocessor().fit(split.X_train)
        scaler = pre.named_transformers_["num"]["scale"]
        np.testing.assert_allclose(scaler.mean_, split.X_train[NUMERIC_FEATURES].mean().to_numpy())

        train_t, test_t = pre.transform(split.X_train), pre.transform(split.X_test)
        assert np.allclose(train_t[[f"num__{c}" for c in NUMERIC_FEATURES]].mean(), 0, atol=1e-9)
        # Test 를 따로 fit 하지 않았으니 Test 평균은 0 과 살짝 다르다 (같으면 Test 로 fit 한 것)
        assert np.abs(test_t[[f"num__{c}" for c in NUMERIC_FEATURES]].mean()).max() > 1e-6

    def test_imputes_missing_values_with_train_statistics(self, split):
        pre = build_preprocessor().fit(split.X_train)
        broken = split.X_test.head(5).copy()
        broken.loc[broken.index[0], "annual_income"] = np.nan
        broken.loc[broken.index[1], "age_group"] = np.nan
        out = pre.transform(broken)
        assert not out.isna().any().any()
        # 결측은 Train 의 중앙값으로 채워지고, 그 값이 표준화되어 들어간다
        median = split.X_train["annual_income"].median()
        scaler = pre.named_transformers_["num"]["scale"]
        expected = (median - scaler.mean_[0]) / scaler.scale_[0]
        assert out.iloc[0]["num__annual_income"] == pytest.approx(expected)

    def test_one_hot_columns(self, split):
        pre = build_preprocessor().fit(split.X_train)
        names = clean_feature_names(pre.get_feature_names_out())
        assert names[: len(NUMERIC_FEATURES)] == NUMERIC_FEATURES
        assert names[len(NUMERIC_FEATURES) :] == [f"{CATEGORICAL_FEATURES[0]}_{g}" for g in AGE_LABELS]

    def test_unseen_category_is_ignored(self, split):
        pre = build_preprocessor().fit(split.X_train)
        odd = split.X_test.head(3).copy()
        odd["age_group"] = "70s"  # Train 에 없던 값
        out = pre.transform(odd)
        assert out[[c for c in out.columns if "age_group" in c]].to_numpy().sum() == 0
