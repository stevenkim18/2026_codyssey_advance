"""오분류 원인을 숫자로 확인한다.

    1. 문서 길이별 정확도: 토큰이 적은 문서는 BoW 벡터에 단서가 거의 없다.
    2. (한국어) 레이블 모호성: KLUE-YNAT는 주석자 3명이 주제를 골랐다. 셋의 의견이 갈린 제목은
       사람도 하나로 정하기 어려운 문서다. 의견 일치/불일치 문서의 정확도를 나눠 보고,
       오분류 중 "모델의 예측을 주석자 중 누군가는 골랐던" 비율을 센다.

실행: `uv run python -m src.error_analysis`
"""

import numpy as np
import pandas as pd

from src.classify import misclassified_table, run_classification
from src.corpus import load_corpus
from src.data import download_ynat


def accuracy_by_length(res, bins: list[int]) -> pd.DataFrame:
    n_tok = res.test["tokens"].map(len).to_numpy()
    correct = res.y_pred == res.test["label"].to_numpy()
    groups = pd.cut(n_tok, bins=bins, right=False)
    out = pd.DataFrame({"length": groups, "correct": correct}).groupby("length", observed=True)["correct"]
    return pd.DataFrame({"docs": out.size(), "accuracy": out.mean()})


def korean_label_ambiguity(res) -> dict:
    raw = pd.read_json(download_ynat()).set_index("guid")
    first_scope = [raw.at[g, "annotations"]["annotations"]["first-scope"] for g in res.test["guid"]]
    unanimous = np.array([len(set(a)) == 1 for a in first_scope])
    correct = res.y_pred == res.test["label"].to_numpy()
    wrong = misclassified_table(res)
    wrong_scope = [first_scope[r] for r in wrong["row"]]
    return {
        "test_docs": len(correct),
        "unanimous_ratio": float(unanimous.mean()),
        "acc_unanimous": float(correct[unanimous].mean()),
        "acc_disagree": float(correct[~unanimous].mean()),
        "errors": len(wrong),
        "error_pred_chosen_by_annotator": float(np.mean([p in s for p, s in zip(wrong["pred"], wrong_scope)])),
    }


def main() -> None:
    en = run_classification(load_corpus("en"), "en", "en")
    print("=== 영어: 문서 길이(토큰 수)별 정확도 ===")
    print(accuracy_by_length(en, [5, 10, 20, 50, 100, 10**6]).to_string(float_format=lambda x: f"{x:.3f}"))

    ko = run_classification(load_corpus("ko"), "ko", "ko-okt")
    print("\n=== 한국어(Okt): 문서 길이(토큰 수)별 정확도 ===")
    print(accuracy_by_length(ko, [2, 5, 7, 9, 11, 100]).to_string(float_format=lambda x: f"{x:.3f}"))

    amb = korean_label_ambiguity(ko)
    print("\n=== 한국어: 주석자 의견 일치 여부와 정확도 (Test) ===")
    print(f"주석자 3명 의견 일치 문서: {amb['unanimous_ratio']:.1%} ({amb['test_docs']:,}건 중)")
    print(f"정확도 - 의견 일치 문서: {amb['acc_unanimous']:.3f} / 의견이 갈린 문서: {amb['acc_disagree']:.3f}")
    print(
        f"오분류 {amb['errors']}건 중 모델 예측을 주석자 중 한 명 이상이 골랐던 비율: "
        f"{amb['error_pred_chosen_by_annotator']:.1%}"
    )


if __name__ == "__main__":
    main()
