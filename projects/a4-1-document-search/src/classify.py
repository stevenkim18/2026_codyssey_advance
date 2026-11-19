"""TF-IDF 벡터 기반 문서 분류 (A4-1): Logistic Regression 기본선.

    - Train/Test 8:2, 카테고리 비율 유지(stratify), random_state=42.
    - TF-IDF(직접 구현)는 **Train 문서로만** fit 한다. Test 문서가 IDF·어휘 사전 계산에 들어가면
      Test 정보가 학습에 새어 들어간다(누수). Test는 Train에서 만든 어휘·IDF로 transform만 한다.
    - 모델: LogisticRegression(C=1.0, max_iter=2000, random_state=42). 다중 클래스는 softmax(multinomial).
    - 지표: 정확도, F1(macro), 카테고리별 precision/recall/F1, 혼동 행렬 그림.
    - 오분류 분석: 틀린 문서마다 "예측 클래스 점수에 가장 크게 기여한 단어"를 계수 × TF-IDF 값으로 뽑는다.
      로지스틱 회귀의 클래스 점수는 Σ_t coef[c, t] · x[t] + b[c] 이므로, 단어별 기여도를 그대로 더하면 점수가 된다.

실행: `uv run python -m src.classify --lang en` / `--lang ko --tokenizer okt|whitespace`
"""

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.corpus import MIN_DF, load_corpus, output_dir
from src.data import KO_LABEL_EN, RANDOM_STATE
from src.tfidf import NumpyTfidf

TEST_SIZE = 0.2


@dataclass
class ClassificationResult:
    name: str
    labels: list[str]
    train: pd.DataFrame
    test: pd.DataFrame
    vectorizer: NumpyTfidf
    model: LogisticRegression
    x_test: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    accuracy: float
    f1_macro: float
    report: str
    cm: np.ndarray


def split(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        corpus, test_size=TEST_SIZE, stratify=corpus["label"], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_classification(
    corpus: pd.DataFrame, lang: str, name: str, tf: str = "raw", c: float = 1.0
) -> ClassificationResult:
    train, test = split(corpus)
    vec = NumpyTfidf(tf=tf, min_df=MIN_DF[lang])
    x_train = vec.fit_transform(train["tokens"].tolist())
    x_test = vec.transform(test["tokens"].tolist())

    model = LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE)
    model.fit(x_train, train["label"])
    y_pred = model.predict(x_test)
    labels = list(model.classes_)
    return ClassificationResult(
        name=name,
        labels=labels,
        train=train,
        test=test,
        vectorizer=vec,
        model=model,
        x_test=x_test,
        y_pred=y_pred,
        proba=model.predict_proba(x_test),
        accuracy=accuracy_score(test["label"], y_pred),
        f1_macro=f1_score(test["label"], y_pred, average="macro"),
        report=classification_report(test["label"], y_pred, digits=3),
        cm=confusion_matrix(test["label"], y_pred, labels=labels),
    )


def display_label(label: str) -> str:
    return KO_LABEL_EN.get(label, label)


def plot_confusion_matrix(res: ClassificationResult, path) -> None:
    cm = res.cm
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    names = [display_label(lab) for lab in res.labels]
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2.5, 1.1 * len(names) + 2))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if row_pct[i, j] > 0.6 else "#222222"
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({res.name})\naccuracy {res.accuracy:.3f}, macro F1 {res.f1_macro:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_contributions(res: ClassificationResult, row: int, label: str, k: int = 6) -> list[tuple[str, float]]:
    """Test 문서 row에서 클래스 label 점수에 기여한 단어 상위 k개: coef[label, t] * x[t]."""
    c = res.labels.index(label)
    contrib = res.model.coef_[c] * res.x_test[row]
    idx = np.argsort(-contrib)[:k]
    return [(res.vectorizer.terms_[j], float(contrib[j])) for j in idx if contrib[j] > 0]


def misclassified_table(res: ClassificationResult) -> pd.DataFrame:
    """틀린 문서 전체: 예측 확신도(예측 클래스 확률) 내림차순."""
    wrong = np.flatnonzero(res.y_pred != res.test["label"].to_numpy())
    rows = []
    for r in wrong:
        pred, true = res.y_pred[r], res.test.at[r, "label"]
        rows.append(
            {
                "row": r,
                "doc_id": res.test.at[r, "doc_id"],
                "true": true,
                "pred": pred,
                "p_pred": res.proba[r, res.labels.index(pred)],
                "p_true": res.proba[r, res.labels.index(true)],
                "n_tokens": len(res.test.at[r, "tokens"]),
            }
        )
    cols = ["row", "doc_id", "true", "pred", "p_pred", "p_true", "n_tokens"]
    return pd.DataFrame(rows, columns=cols).sort_values("p_pred", ascending=False).reset_index(drop=True)


def describe_case(res: ClassificationResult, r: int, text_len: int = 300) -> str:
    pred, true = res.y_pred[r], res.test.at[r, "label"]
    text = " ".join(res.test.at[r, "text"].split())
    pred_terms = ", ".join(f"{t}({v:+.2f})" for t, v in top_contributions(res, r, pred))
    true_terms = ", ".join(f"{t}({v:+.2f})" for t, v in top_contributions(res, r, true)) or "(없음)"
    return "\n".join(
        [
            (
                f"[{res.test.at[r, 'doc_id']}] 예측: {pred} ({res.proba[r, res.labels.index(pred)]:.2f}) / "
                f"실제: {true} ({res.proba[r, res.labels.index(true)]:.2f}), 토큰 {len(res.test.at[r, 'tokens'])}개"
            ),
            f"  문서: \"{text[:text_len]}{'…' if len(text) > text_len else ''}\"",
            f"  예측 클래스 쪽 기여 단어: {pred_terms}",
            f"  실제 클래스 쪽 기여 단어: {true_terms}",
        ]
    )


def summary_text(res: ClassificationResult) -> str:
    return "\n".join(
        [
            "=== 분류 성능 평가 ===",
            f"데이터: {res.name}",
            (
                "모델: LogisticRegression(C=1.0, max_iter=2000, random_state=42), 특성: 직접 구현 TF-IDF "
                f"(tf={res.vectorizer.tf}, smooth_idf=True, l2, min_df={res.vectorizer.min_df}, Train에서만 fit)"
            ),
            f"분할: Train {len(res.train):,} / Test {len(res.test):,} (8:2, stratify, random_state={RANDOM_STATE})",
            f"어휘 크기(Train 기준): {len(res.vectorizer.vocabulary_):,}",
            "",
            f"정확도: {res.accuracy:.3f}",
            f"F1-Score (macro): {res.f1_macro:.3f}",
            "",
            res.report,
        ]
    )


def run(lang: str, tokenizer: str = "okt", n_cases: int = 8) -> ClassificationResult:
    name = lang if lang == "en" else f"{lang}-{tokenizer}"
    corpus = load_corpus(lang, tokenizer=tokenizer)
    res = run_classification(corpus, lang, name)
    out = output_dir(lang, tokenizer)

    plot_confusion_matrix(res, out / "confusion_matrix.png")
    wrong = misclassified_table(res)
    cases = "\n\n".join(describe_case(res, r) for r in wrong["row"])
    (out / "classification_report.txt").write_text(summary_text(res) + "\n", encoding="utf-8")
    (out / "misclassified.txt").write_text(
        f"=== 오분류 {len(wrong)}건 (예측 확신도 높은 순) ===\n\n{cases}\n", encoding="utf-8"
    )

    print(summary_text(res))
    print(f"=== 오분류 상위 {n_cases}건 (확신도 높은 순, 전체 {len(wrong)}건은 misclassified.txt) ===\n")
    for r in wrong["row"][:n_cases]:
        print(describe_case(res, r), "\n")
    print(f"→ 저장: {out.name}/confusion_matrix.png, classification_report.txt, misclassified.txt")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--tokenizer", choices=["okt", "whitespace"], default="okt")
    args = ap.parse_args()
    run(args.lang, args.tokenizer)
