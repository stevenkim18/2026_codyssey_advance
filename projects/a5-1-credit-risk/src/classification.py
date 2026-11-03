"""연체 위험 예측 분류 모델 (A5-1): 불균형 처리, 랜덤 포레스트 튜닝, 베이스라인과의 비교.

구성
    1. 불균형 처리 비교: 같은 로지스틱 회귀를 (처리 없음 / class_weight='balanced' / SMOTE) 로 학습한다.
       SMOTE 는 imblearn Pipeline 안에 넣어서 교차 검증의 Train fold 에서만 합성 샘플을 만든다.
    2. 랜덤 포레스트 + GridSearchCV: 하이퍼파라미터를 5-fold 교차 검증 F1 로 고른다.
       후보는 3 x 5 x 4 = 60 조합 (파라미터당 5개 이하, 총 100개 이하).
    3. 최종 비교: 규칙 기반 베이스라인, 로지스틱 회귀, 랜덤 포레스트(기본/튜닝) 를 Test 에서 비교한다.
    4. 그래프: Confusion Matrix, ROC 곡선(AUC), 특징 중요도.
    5. 누수 실험: `credit_score` 를 feature 에 넣으면 성능이 얼마나 부풀려지는지 확인한다.

연체 여부는 `credit_score` 하위 15% 에서 만들어진 값이라 `credit_score` 는 입력에 쓰지 않는다.
모든 모델은 Test 를 한 번도 보지 않은 채 Train 으로만 학습·선택하고, 확률 0.5 를 기준으로 연체를 판정한다.

실행: `uv run python -m src.classification` (프로젝트 폴더에서, 약 1~3분)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.baseline import run_baseline
from src.metrics import classification_metrics, to_markdown_table
from src.plotting import (
    AXIS,
    BLUE,
    CATEGORICAL,
    INK,
    MUTED,
    SEQ_BLUE,
    apply_style,
    save_figure,
)
from src.preprocess import (
    NUMERIC_FEATURES,
    RANDOM_STATE,
    Split,
    build_preprocessor,
    clean_feature_names,
    load_data,
    make_cv,
    split_data,
)

# GridSearchCV 탐색 공간: 파라미터당 후보 5개 이하, 총 조합 100개 이하
PARAM_GRID = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [4, 6, 8, 12, None],
    "rf__min_samples_leaf": [1, 5, 10, 20],
}
SCORING = {"f1": "f1", "auc": "roc_auc"}  # 고르는 기준은 F1 (refit), AUC 는 함께 기록

LOGREG_VARIANTS = {
    "none": "LogReg (처리 없음)",
    "balanced": "LogReg (class_weight=balanced)",
    "smote": "LogReg (SMOTE)",
}


def count_grid_combinations(grid: dict) -> int:
    """GridSearchCV 가 시도하는 조합의 수."""
    return int(np.prod([len(values) for values in grid.values()]))


def make_logreg(handling: str) -> ImbPipeline:
    """전처리 → (SMOTE) → 로지스틱 회귀. handling: 'none' | 'balanced' | 'smote'."""
    if handling not in LOGREG_VARIANTS:
        raise ValueError(f"handling 은 {list(LOGREG_VARIANTS)} 중 하나여야 합니다: {handling}")
    steps = [("pre", build_preprocessor())]
    if handling == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    class_weight = "balanced" if handling == "balanced" else None
    steps.append(("clf", LogisticRegression(class_weight=class_weight, max_iter=1000)))
    return ImbPipeline(steps)


def make_forest(**params) -> Pipeline:
    """전처리 → 랜덤 포레스트 (class_weight='balanced'). params 는 RandomForestClassifier 인자."""
    forest = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, **params)
    return Pipeline([("pre", build_preprocessor()), ("rf", forest)])


def evaluate(model, X, y) -> dict:
    """연체 확률 0.5 기준의 예측과 점수를 함께 돌려준다."""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {"y_pred": y_pred, "y_score": y_score, "metrics": classification_metrics(y, y_pred, y_score)}


def tune_forest(split: Split) -> GridSearchCV:
    """랜덤 포레스트를 GridSearchCV 로 튜닝한다. 전처리가 Pipeline 안에 있어 fold 마다 다시 fit 된다."""
    search = GridSearchCV(
        make_forest(),
        PARAM_GRID,
        scoring=SCORING,
        refit="f1",
        cv=make_cv(),
        n_jobs=-1,
    )
    return search.fit(split.X_train, split.y_clf_train)


def leakage_experiment(split: Split) -> pd.DataFrame:
    """`credit_score` 를 feature 로 넣었을 때와 뺐을 때를 비교한다 (랜덤 포레스트 기본 설정)."""
    rows = {}
    for label, leak in (("credit_score 제외 (정상)", False), ("credit_score 포함 (누수)", True)):
        X_train, X_test, numeric = split.X_train, split.X_test, NUMERIC_FEATURES
        if leak:
            X_train = X_train.assign(credit_score=split.y_reg_train)
            X_test = X_test.assign(credit_score=split.y_reg_test)
            numeric = [*NUMERIC_FEATURES, "credit_score"]
        forest = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        model = Pipeline([("pre", build_preprocessor(numeric)), ("rf", forest)])
        model.fit(X_train, split.y_clf_train)
        rows[label] = evaluate(model, X_test, split.y_clf_test)["metrics"]
    return pd.DataFrame(rows).T[["Accuracy", "Recall", "F1-Score", "AUC"]].rename_axis("feature")


def impurity_importance(model: Pipeline) -> pd.Series:
    """랜덤 포레스트의 불순도 기반 중요도. 원-핫으로 나뉜 `age_group_*` 는 하나로 합친다."""
    names = clean_feature_names(model[0].get_feature_names_out())
    imp = pd.Series(model[-1].feature_importances_, index=names)
    imp.index = ["age_group" if n.startswith("age_group_") else n for n in imp.index]
    return imp.groupby(level=0).sum().sort_values(ascending=False)


def plot_confusion_matrices(panels: dict[str, tuple[np.ndarray, np.ndarray]]):
    """모델별 Confusion Matrix. 색은 '실제 클래스 안에서의 비율'이고 칸에는 건수를 적는다.

    연체가 12% 뿐이라 건수로 색을 칠하면 연체 칸이 전부 연하게 보인다. 행 비율로 칠하면
    '실제 연체 중 몇 %를 잡았는가(Recall)' 가 대각선 색으로 바로 읽힌다.
    """
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.3))
    for ax, (name, (y_true, y_pred)) in zip(np.atleast_1d(axes), panels.items(), strict=True):
        cm = confusion_matrix(y_true, y_pred)
        rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({rate[i, j]:.0%})",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if rate[i, j] > 0.45 else INK,
                )
        ax.set_xticks([0, 1], ["Normal (0)", "Overdue (1)"])
        ax.set_yticks([0, 1], ["Normal (0)", "Overdue (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]):
    """모델별 ROC 곡선. 범례에 AUC 를 적는다. 대각선은 무작위 추측(AUC 0.5)이다."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, zorder=1, label="Random (AUC 0.500)")
    for (name, (fpr, tpr, auc)), color in zip(curves.items(), CATEGORICAL, strict=False):
        points = len(fpr) <= 12  # 규칙 기반은 점수가 6단계뿐이라 꺾은선에 점을 찍어 구분한다
        ax.plot(
            fpr,
            tpr,
            color=color,
            zorder=3,
            marker="o" if points else None,
            markersize=5,
            label=f"{name} (AUC {auc:.3f})",
        )
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_feature_importance(impurity: pd.Series, permutation: pd.Series):
    """왼쪽: 불순도 기반(Train), 오른쪽: 순열 중요도(Test, 변수를 섞었을 때 AUC 감소량)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    panels = (
        (axes[0], impurity, "Impurity-based (train)", "Importance (sums to 1)"),
        (axes[1], permutation, "Permutation (test)", "AUC drop when shuffled"),
    )
    for ax, series, title, xlabel in panels:
        series = series.sort_values()
        ax.barh(series.index, series.values, color=BLUE, height=0.6, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    split = split_data(load_data())
    y_test = split.y_clf_test
    print(
        f"Train {len(split.X_train):,}건 (연체 {int(split.y_clf_train.sum())}), "
        f"Test {len(split.X_test):,}건 (연체 {int(y_test.sum())})\n"
    )

    # 1. 불균형 처리 비교
    logregs = {k: make_logreg(k).fit(split.X_train, split.y_clf_train) for k in LOGREG_VARIANTS}
    logreg_results = {name: evaluate(logregs[k], split.X_test, y_test) for k, name in LOGREG_VARIANTS.items()}
    print("[불균형 처리 비교] 로지스틱 회귀, Test, 확률 0.5 기준")
    table = pd.DataFrame({n: r["metrics"] for n, r in logreg_results.items()}).T.rename_axis("모델")
    print(to_markdown_table(table))

    # 2. 랜덤 포레스트 튜닝
    n_combos = count_grid_combinations(PARAM_GRID)
    print(f"\n[GridSearchCV] {n_combos}개 조합 x {make_cv().get_n_splits()}-fold, 기준 F1 ...")
    start = time.time()
    search = tune_forest(split)
    print(f"완료 ({time.time() - start:.0f}초)")
    best_params = {k.removeprefix("rf__"): v for k, v in search.best_params_.items()}
    print(f"최적 파라미터: {best_params}")
    cv = pd.DataFrame(search.cv_results_)
    top = cv.sort_values("rank_test_f1").head(5)
    top_table = pd.DataFrame(
        {
            "n_estimators": top["param_rf__n_estimators"].astype(int),
            "max_depth": top["param_rf__max_depth"].map(lambda v: "None" if pd.isna(v) else int(v)),
            "min_samples_leaf": top["param_rf__min_samples_leaf"].astype(int),
            "CV F1": top["mean_test_f1"],
            "CV AUC": top["mean_test_auc"],
        }
    ).reset_index(drop=True)
    top_table.index = top_table.index + 1
    print("\n[CV 상위 5개 조합]")
    print(to_markdown_table(top_table.rename_axis("순위")))

    forest_tuned = search.best_estimator_
    forest_default = make_forest(n_estimators=100, n_jobs=-1).fit(split.X_train, split.y_clf_train)

    # Train 5-fold 교차 검증으로 같은 fold 에서 비교한다. Test 는 연체가 240건뿐이라 작은 차이는 우연일 수 있다.
    cv_splits = list(make_cv().split(split.X_train, split.y_clf_train))
    cv_rows = {}
    for name, model in (
        (LOGREG_VARIANTS["balanced"], make_logreg("balanced")),
        ("Random Forest (기본)", make_forest(n_estimators=100, n_jobs=-1)),
        ("Random Forest (튜닝)", make_forest(n_jobs=-1, **best_params)),
    ):
        cv_score = cross_validate(
            model, split.X_train, split.y_clf_train, cv=cv_splits, scoring=SCORING
        )
        cv_rows[name] = {
            "CV F1": cv_score["test_f1"].mean(),
            "CV F1 표준편차": cv_score["test_f1"].std(),
            "CV AUC": cv_score["test_auc"].mean(),
        }
    print("\n[Train 5-fold 교차 검증 비교] 같은 fold, 확률 0.5 기준")
    print(to_markdown_table(pd.DataFrame(cv_rows).T.rename_axis("모델")))

    # 3. 최종 비교 (Test)
    baseline = run_baseline(split)
    results = {
        "규칙 기반 베이스라인": {"y_pred": baseline["y_pred"], "y_score": baseline["y_score"], "metrics": baseline["metrics"]},
        "Logistic Regression": logreg_results[LOGREG_VARIANTS["balanced"]],
        "Random Forest (기본)": evaluate(forest_default, split.X_test, y_test),
        "Random Forest (튜닝)": evaluate(forest_tuned, split.X_test, y_test),
    }
    print("\n[최종 성능 비교] Test, 연체(1)가 양성")
    final = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.rename_axis("모델")
    print(to_markdown_table(final))
    base_f1, base_auc = final.loc["규칙 기반 베이스라인", ["F1-Score", "AUC"]]
    tuned_f1, tuned_auc = final.loc["Random Forest (튜닝)", ["F1-Score", "AUC"]]
    print(
        f"\n튜닝 RF vs 베이스라인: F1 {tuned_f1 / base_f1 - 1:+.1%}, AUC {tuned_auc / base_auc - 1:+.1%}"
    )

    # 4. 특징 중요도
    impurity = impurity_importance(forest_tuned)
    perm = permutation_importance(
        forest_tuned, split.X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    permutation = pd.Series(perm.importances_mean, index=split.X_test.columns)
    importance = pd.DataFrame({"불순도 기반": impurity, "순열(AUC 감소)": permutation}).sort_values(
        "불순도 기반", ascending=False
    )
    print("\n[특징 중요도] 튜닝된 랜덤 포레스트")
    print(to_markdown_table(importance.rename_axis("변수"), floatfmt=".4f"))

    # 5. 누수 실험
    print("\n[데이터 누수 실험] 랜덤 포레스트 기본 설정, Test")
    print(to_markdown_table(leakage_experiment(split), floatfmt=".3f"))

    # 6. 그래프
    names = ["규칙 기반 베이스라인", "Logistic Regression", "Random Forest (튜닝)"]
    english = {
        "규칙 기반 베이스라인": "Rule-based baseline",
        "Logistic Regression": "Logistic Regression",
        "Random Forest (튜닝)": "Random Forest (tuned)",
    }
    panels = {
        f"{english[n]}  (F1 {results[n]['metrics']['F1-Score']:.2f})": (y_test.to_numpy(), results[n]["y_pred"])
        for n in names
    }
    curves = {}
    for n in names:
        fpr, tpr, _ = roc_curve(y_test, results[n]["y_score"])
        curves[english[n]] = (fpr, tpr, results[n]["metrics"]["AUC"])
    saved = [
        save_figure(plot_confusion_matrices(panels), "classification_confusion_matrix.png"),
        save_figure(plot_roc_curves(curves), "classification_roc_curve.png"),
        save_figure(plot_feature_importance(impurity, permutation), "classification_feature_importance.png"),
    ]
    print("\n그래프 저장: " + ", ".join(p.name for p in saved))


if __name__ == "__main__":
    main()
