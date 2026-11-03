"""신용 점수 예측 회귀 모델 (A5-1): Ridge 와 Lasso, 규제 강도(Alpha)별 계수 변화.

구성
    1. Ridge(L2) / Lasso(L1) 를 Alpha ∈ {0.01, 0.1, 1, 10, 100} 로 각각 학습한다.
       전처리(대치 → 스케일링 → 인코딩)는 Pipeline 안에 있어서 Train 으로만 fit 되고,
       5-fold 교차 검증에서도 fold 마다 Train 부분으로 다시 fit 된다.
    2. Alpha 는 교차 검증 RMSE 가 가장 낮은 값으로 고른다. Test 는 고른 뒤 보고용으로만 쓴다.
    3. RMSE · MAE · R² 3개를 Test 에서 잰다. 예측값은 0~1000 으로 자른다.
    4. 그래프: Alpha 별 계수 변화, 예측값 대 실제값.

계수는 표준화된 변수 기준이므로 "변수가 1 표준편차 오를 때 신용 점수가 몇 점 바뀌는가" 이다.
그래서 단위가 다른 변수(소득은 수천, 부채 비율은 0~1)의 계수 크기를 직접 비교할 수 있다.

실행: `uv run python -m src.regression` (프로젝트 폴더에서)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.metrics import regression_metrics, to_markdown_table
from src.plotting import (
    BLUE,
    CATEGORICAL,
    INK_SECONDARY,
    MARKERS,
    MUTED,
    ORANGE,
    apply_style,
    save_figure,
)
from src.preprocess import (
    Split,
    build_preprocessor,
    clean_feature_names,
    load_data,
    make_cv,
    split_data,
)

ALPHAS = [0.01, 0.1, 1, 10, 100]
SCORE_MIN, SCORE_MAX = 0, 1000
ZERO_TOL = 1e-8  # 이보다 작은 계수는 0 으로 본다

# 데이터 생성식에 실제로 들어 있는 변수 (나머지는 신용 점수에 영향이 없다)
SIGNAL_FEATURES = ["annual_income", "debt_ratio", "overdue_count_6m"]


def make_model(kind: str, alpha: float) -> Pipeline:
    """전처리 + 회귀. kind 는 'Ridge' 또는 'Lasso'."""
    if kind == "Ridge":
        reg = Ridge(alpha=alpha)
    elif kind == "Lasso":
        reg = Lasso(alpha=alpha, max_iter=20000)
    else:
        raise ValueError(f"kind 는 'Ridge' 또는 'Lasso' 여야 합니다: {kind}")
    return Pipeline([("pre", build_preprocessor()), ("model", reg)])


def predict_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """신용 점수 예측값. 점수는 0~1000 범위이므로 벗어난 값은 잘라낸다."""
    return np.clip(model.predict(X), SCORE_MIN, SCORE_MAX)


def run_regression(split: Split) -> dict:
    """두 모델 x Alpha 5개를 학습·평가한다. 결과표, 계수표, 학습된 모델을 돌려준다."""
    cv_splits = list(make_cv().split(split.X_train, split.y_clf_train))  # 연체 여부로 층화한 fold
    rows, coefs, models = [], {}, {}

    for kind in ("Ridge", "Lasso"):
        for alpha in ALPHAS:
            model = make_model(kind, alpha).fit(split.X_train, split.y_reg_train)
            cv = cross_validate(
                make_model(kind, alpha),
                split.X_train,
                split.y_reg_train,
                cv=cv_splits,
                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            )
            test = regression_metrics(split.y_reg_test, predict_score(model, split.X_test))
            coef = pd.Series(
                model[-1].coef_, index=clean_feature_names(model[0].get_feature_names_out())
            )
            rows.append(
                {
                    "모델": kind,
                    "Alpha": alpha,
                    "CV RMSE": -cv["test_rmse"].mean(),
                    "CV R2": cv["test_r2"].mean(),
                    "RMSE": test["RMSE"],
                    "MAE": test["MAE"],
                    "R2": test["R2"],
                    "0인 계수": int((coef.abs() < ZERO_TOL).sum()),
                }
            )
            coefs[(kind, alpha)] = coef
            models[(kind, alpha)] = model

    results = pd.DataFrame(rows)
    # Alpha 선택은 교차 검증 RMSE 로 (Test 를 보고 고르지 않는다)
    best = {
        kind: float(results[results["모델"] == kind].nsmallest(1, "CV RMSE")["Alpha"].iloc[0])
        for kind in ("Ridge", "Lasso")
    }
    return {"results": results, "coefs": coefs, "models": models, "best_alpha": best}


def coefficient_table(coefs: dict, kind: str) -> pd.DataFrame:
    """행 = 변수, 열 = Alpha 인 계수표."""
    table = pd.DataFrame({alpha: coefs[(kind, alpha)] for alpha in ALPHAS})
    table.columns = [f"α={a:g}" for a in ALPHAS]
    return table.rename_axis("변수")


def plot_coefficient_path(coefs: dict):
    """Ridge / Lasso 의 Alpha 별 계수 변화. 실제로 영향이 있는 3개 변수만 색을 입히고 나머지는 회색."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    signal_colors = dict(zip(SIGNAL_FEATURES, CATEGORICAL, strict=True))
    signal_markers = dict(zip(SIGNAL_FEATURES, MARKERS, strict=True))
    label_offset = {"annual_income": (6, 5), "debt_ratio": (6, 5), "overdue_count_6m": (6, -12)}

    for ax, kind in zip(axes, ("Ridge", "Lasso"), strict=True):
        table = pd.DataFrame({a: coefs[(kind, a)] for a in ALPHAS})
        for feature, series in table.iterrows():
            if feature in SIGNAL_FEATURES:
                continue
            ax.plot(ALPHAS, series.values, color=MUTED, linewidth=1.0, alpha=0.8, zorder=2)
        for feature in SIGNAL_FEATURES:
            ax.plot(
                ALPHAS,
                table.loc[feature].values,
                color=signal_colors[feature],
                marker=signal_markers[feature],
                markersize=5,
                zorder=3,
                label=feature,
            )
            ax.annotate(
                feature,
                (ALPHAS[0], table.loc[feature].iloc[0]),
                xytext=label_offset[feature],
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=9,
            )
        ax.plot([], [], color=MUTED, linewidth=1.0, label="other 7 (spending, cards, age groups)")
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ax.minorticks_off()
        ax.set_xlabel("Alpha (log scale)")
        ax.set_title(f"{kind}: coefficients vs. regularization")
    axes[0].set_ylabel("Coefficient (score points per 1 SD)")
    axes[0].set_ylim(-50, 68)  # 위아래 직접 라벨이 축에 걸리지 않도록 여유를 둔다 (sharey)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(model: Pipeline, split: Split, title: str):
    """Test 세트의 예측 점수 대 실제 점수. 점이 대각선에 붙을수록 정확하다."""
    pred = predict_score(model, split.X_test)
    actual = split.y_reg_test.to_numpy()
    m = regression_metrics(actual, pred)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(actual, pred, s=10, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
    lo, hi = min(actual.min(), pred.min()) - 15, max(actual.max(), pred.max()) + 15
    ax.plot([lo, hi], [lo, hi], color=ORANGE, linewidth=2.0, zorder=3, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual credit score")
    ax.set_ylabel("Predicted credit score")
    ax.set_title(title)
    ax.text(
        0.04,
        0.95,
        f"RMSE {m['RMSE']:.1f}  MAE {m['MAE']:.1f}  R² {m['R2']:.3f}",
        transform=ax.transAxes,
        va="top",
        color=INK_SECONDARY,
        fontsize=10,
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    split = split_data(load_data())
    out = run_regression(split)
    results, coefs, best = out["results"], out["coefs"], out["best_alpha"]

    labeled = results.assign(
        모델=results["모델"] + " (α=" + results["Alpha"].map("{:g}".format) + ")"
    ).set_index("모델")

    print("[Alpha 별 성능] CV = Train 5-fold, 나머지는 Test (예측값은 0~1000 으로 clip)")
    print(to_markdown_table(labeled.drop(columns="Alpha")))
    print(f"\nCV RMSE 기준 최적 Alpha: Ridge {best['Ridge']:g}, Lasso {best['Lasso']:g}")

    print("\n[회귀 성능 비교표] 최적 Alpha, Test")
    best_names = [f"{kind} (α={alpha:g})" for kind, alpha in best.items()]
    print(to_markdown_table(labeled.loc[best_names, ["RMSE", "MAE", "R2"]]))

    for kind in ("Ridge", "Lasso"):
        print(f"\n[{kind} 계수] 표준화된 변수 기준")
        print(to_markdown_table(coefficient_table(coefs, kind), floatfmt=".2f"))

    p1 = save_figure(plot_coefficient_path(coefs), "regression_coef_path.png")
    best_model = out["models"][("Ridge", best["Ridge"])]
    p2 = save_figure(
        plot_pred_vs_actual(best_model, split, f"Ridge (alpha={best['Ridge']:g}): Test predictions"),
        "regression_pred_vs_actual.png",
    )
    print(f"\n그래프 저장: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
