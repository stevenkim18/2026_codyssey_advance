import numpy as np
import pandas as pd
import pytest
from src.pipeline import DataAnalyzer


def make_analyzer(df: pd.DataFrame) -> DataAnalyzer:
    analyzer = DataAnalyzer()
    analyzer.df = df
    return analyzer


# ---------------------------------------------------------------- 이상치
def test_iqr_bounds_and_detect_outliers():
    values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]
    analyzer = make_analyzer(pd.DataFrame({"x": values}))

    lower, upper = analyzer.iqr_bounds("x", threshold=1.5)
    q1, q3 = np.percentile(values, [25, 75])
    assert lower == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert upper == pytest.approx(q3 + 1.5 * (q3 - q1))

    outliers = analyzer.detect_outliers("x")
    assert outliers["x"].tolist() == [1000]


def test_larger_threshold_finds_fewer_outliers():
    rng = np.random.default_rng(0)
    analyzer = make_analyzer(pd.DataFrame({"x": rng.normal(0, 1, 500)}))
    assert len(analyzer.detect_outliers("x", 3.0)) <= len(
        analyzer.detect_outliers("x", 1.0)
    )


def test_treat_outliers_clip_and_remove():
    analyzer = make_analyzer(
        pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    )
    _, upper = analyzer.iqr_bounds("x")

    clipped = analyzer.treat_outliers("x", method="clip")
    assert len(clipped) == 10 and clipped["x"].max() == pytest.approx(upper)

    removed = analyzer.treat_outliers("x", method="remove")
    assert len(removed) == 9 and removed["x"].max() == 18

    assert analyzer.df["x"].max() == 1000  # 원본은 그대로
    with pytest.raises(ValueError):
        analyzer.treat_outliers("x", method="bad")


# ---------------------------------------------------------------- 결측치
def test_group_median_fill_uses_group_statistics():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1.0, 3.0, np.nan, 10.0, 30.0, np.nan],
        }
    )
    analyzer = make_analyzer(df)
    analyzer.handle_missing_values("x", "group_median", "g")
    assert analyzer.df["x"].tolist() == [1.0, 3.0, 2.0, 10.0, 30.0, 20.0]


def test_group_fill_falls_back_to_overall_statistic():
    # 그룹 c는 값이 전부 결측, 마지막 행은 그룹 자체가 결측
    df = pd.DataFrame(
        {"g": ["a", "a", "c", "c", None], "x": [2.0, 4.0, np.nan, np.nan, np.nan]}
    )
    analyzer = make_analyzer(df)
    analyzer.handle_missing_values("x", "group_mean", "g")
    assert analyzer.df["x"].tolist() == [2.0, 4.0, 3.0, 3.0, 3.0]


def test_unit_col_computes_statistics_per_unit_not_per_row():
    # 상품 p1은 3행(값 100), p2는 1행(값 0) -> 행 기준 평균 75, 상품 기준 평균 50
    df = pd.DataFrame(
        {
            "g": ["a"] * 5,
            "unit": ["p1", "p1", "p1", "p2", "p3"],
            "x": [100.0, 100.0, 100.0, 0.0, np.nan],
        }
    )
    by_row = make_analyzer(df.copy())
    by_row.handle_missing_values("x", "group_mean", "g")
    by_unit = make_analyzer(df.copy())
    by_unit.handle_missing_values("x", "group_mean", "g", unit_col="unit")
    assert by_row.df["x"].iloc[-1] == pytest.approx(75.0)
    assert by_unit.df["x"].iloc[-1] == pytest.approx(50.0)


def test_other_missing_strategies_and_errors():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 9.0]})
    mean = make_analyzer(df.copy())
    mean.handle_missing_values("x", "mean")
    assert mean.df["x"].iloc[2] == pytest.approx(4.0)

    dropped = make_analyzer(df.copy())
    dropped.handle_missing_values("x", "drop")
    assert len(dropped.df) == 3

    with pytest.raises(ValueError):
        make_analyzer(df.copy()).handle_missing_values(
            "x", "group_mean"
        )  # group_col 없음
    with pytest.raises(ValueError):
        make_analyzer(df.copy()).handle_missing_values("x", "nope")


# ---------------------------------------------------------------- 피처
def test_text_features():
    analyzer = make_analyzer(pd.DataFrame({"prod_name": ["Red T-Shirt", "Hat", None]}))
    analyzer.add_text_features("prod_name")
    assert analyzer.df["prod_name_word_count"].iloc[:2].tolist() == [2, 1]
    assert analyzer.df["prod_name_len"].iloc[:2].tolist() == [11, 3]
    assert pd.isna(analyzer.df["prod_name_word_count"].iloc[2])


def test_image_features_match_per_image_loop_and_flag_missing_images():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(4, 6, 5, 3), dtype=np.uint8)
    ids = ["a1", "a2", "a3", "a4"]
    df = pd.DataFrame({"article_id": ["a1", "a2", "a2", "zz"]})
    analyzer = make_analyzer(df)
    analyzer.add_image_features(images, ids)

    out = analyzer.df
    assert out["img_mean"].iloc[0] == pytest.approx(images[0].mean())
    assert out["img_std"].iloc[1] == pytest.approx(images[1].std())
    assert out["has_image"].tolist() == [True, True, True, False]
    assert pd.isna(out["img_mean"].iloc[3])
    assert len(out) == len(df)  # merge로 행이 늘지 않는다


def test_add_image_features_is_idempotent():
    images = np.ones((1, 2, 2, 3), dtype=np.uint8)
    analyzer = make_analyzer(pd.DataFrame({"article_id": ["a1"]}))
    analyzer.add_image_features(images, ["a1"])
    analyzer.add_image_features(images, ["a1"])
    assert {"img_mean", "img_std", "has_image"} <= set(analyzer.df.columns)
    assert not any(c.endswith(("_x", "_y")) for c in analyzer.df.columns)


def test_image_features_reject_wrong_dimensions():
    analyzer = make_analyzer(pd.DataFrame({"article_id": ["a1"]}))
    with pytest.raises(ValueError):
        analyzer.add_image_features(np.zeros((1, 4, 4)), ["a1"])


# ---------------------------------------------------------------- 통계
def test_summary_statistics_and_correlation():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "z": [4.0, 3.0, 2.0, 1.0],
        }
    )
    analyzer = make_analyzer(df)
    stats = analyzer.summary_statistics(["x"])
    assert stats.loc["x", "mean"] == pytest.approx(2.5)
    assert stats.loc["x", "iqr"] == pytest.approx(
        stats.loc["x", "q3"] - stats.loc["x", "q1"]
    )

    corr = analyzer.correlation_pairs([("x", "y"), ("x", "z")])
    assert corr["r"].tolist() == pytest.approx([1.0, -1.0])


# ---------------------------------------------------------------- RFM
def rfm_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B", "B", "C", "D", "D"],
            "t_dat": pd.to_datetime(
                [
                    "2025-12-20",
                    "2025-12-25",
                    "2025-12-30",
                    "2025-10-01",
                    "2025-11-15",
                    "2025-06-01",
                    "2025-12-31",
                    "2025-12-31",
                ]
            ),
            "amount": [50.0, 70.0, 80.0, 30.0, 20.0, 15.0, 100.0, 120.0],
        }
    )


def test_calculate_rfm_values():
    analyzer = make_analyzer(rfm_fixture())
    rfm = analyzer.calculate_rfm()

    # 기준일 = 마지막 거래일(2025-12-31) + 1일 = 2026-01-01
    assert rfm.loc["A", "recency"] == 2
    assert rfm.loc["D", "recency"] == 1
    assert rfm.loc["C", "recency"] == 214
    # 같은 날 두 번 산 D의 frequency는 1 (서로 다른 날짜 수)
    assert rfm.loc["D", "frequency"] == 1
    assert rfm.loc["A", "frequency"] == 3
    assert rfm.loc["A", "monetary"] == pytest.approx(200.0)


def test_rfm_recency_score_is_reversed():
    rfm = make_analyzer(rfm_fixture()).calculate_rfm()
    assert rfm.loc["D", "R"] > rfm.loc["C", "R"]  # 최근일수록 높은 점수
    assert rfm["R"].between(1, 5).all()


def test_rfm_same_value_gets_same_score():
    df = pd.DataFrame(
        {
            "customer_id": list("ABCDEFGH"),
            "t_dat": pd.to_datetime(["2025-01-01"] * 8),
            "amount": [10.0] * 8,
        }
    )
    rfm = make_analyzer(df).calculate_rfm()
    assert rfm["M"].nunique() == 1 and rfm["F"].nunique() == 1


def test_rfm_custom_snapshot_and_segments_exist():
    analyzer = make_analyzer(rfm_fixture())
    rfm = analyzer.calculate_rfm(snapshot_date="2026-01-31")
    assert rfm.loc["D", "recency"] == 31
    assert set(rfm["segment"]) <= {
        "VIP",
        "Loyal",
        "New",
        "At Risk",
        "Churned",
        "Regular",
    }

    summary = analyzer.segment_summary()
    assert summary["customers"].sum() == 4
    assert summary["customer_pct"].sum() == pytest.approx(100.0)
    assert summary["revenue_pct"].sum() == pytest.approx(100.0)


def test_methods_require_load_data_first():
    with pytest.raises(RuntimeError):
        DataAnalyzer().calculate_rfm()
