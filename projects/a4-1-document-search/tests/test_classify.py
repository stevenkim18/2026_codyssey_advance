"""분류: 분할 비율·층화, TF-IDF 를 Train 으로만 fit 하는지 (가짜 말뭉치)."""

import numpy as np
import pandas as pd
from src.classify import misclassified_table, run_classification, split


def _corpus(n_per_class=50):
    rng = np.random.default_rng(0)
    vocab = {"space": ["orbit", "nasa", "launch", "moon"], "sport": ["goal", "team", "season", "win"]}
    rows = []
    for label, words in vocab.items():
        for _ in range(n_per_class):
            rows.append({"text": "", "label": label, "tokens": list(rng.choice(words + ["common", "word"], size=8))})
    rows.append({"text": "", "label": "space", "tokens": ["orbit"] * 3})
    df = pd.DataFrame(rows)
    df.insert(0, "doc_id", [f"t{i}" for i in range(len(df))])
    return df


def test_split_8_2_stratified():
    train, test = split(_corpus())
    assert len(train) == 80 and len(test) == 21  # 101건의 20% → 올림 21
    assert abs(test["label"].value_counts()["space"] - test["label"].value_counts()["sport"]) <= 1


def test_vectorizer_fit_on_train_only():
    corpus = _corpus()
    _, test = split(corpus)
    target = test.at[0, "doc_id"]
    corpus["tokens"] = [
        ["onlyintest"] * 3 + ["orbit"] if d == target else toks for d, toks in zip(corpus["doc_id"], corpus["tokens"])
    ]
    res = run_classification(corpus, "en", "toy")
    assert "onlyintest" not in res.vectorizer.vocabulary_
    assert res.vectorizer.n_docs_ == len(res.train)


def test_metrics_and_misclassified_table():
    res = run_classification(_corpus(), "en", "toy")
    assert 0.0 <= res.f1_macro <= 1.0 and res.cm.sum() == len(res.test)
    table = misclassified_table(res)
    assert len(table) == (res.y_pred != res.test["label"].to_numpy()).sum()
