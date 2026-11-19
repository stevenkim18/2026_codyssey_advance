"""불용어 후보 찾기: 문서 빈도(DF)가 높으면서 카테고리 사이에 고르게 퍼진 단어.

두 기준을 함께 본다.
    - DF 비율 = (단어가 나온 문서 수) / (전체 문서 수). 높을수록 IDF가 작아 TF-IDF 가중치가 이미 낮다.
    - 카테고리 균등도 = 카테고리별 출현 비율 분포의 정규화 엔트로피 (0~1).
      1에 가까우면 어느 카테고리에서나 비슷하게 나와 주제를 가르지 못한다.
DF가 높아도 균등도가 낮은 단어(예: 'game'은 야구·하키에 몰림)는 주제 신호이므로 불용어로 쓰지 않는다.

실행: `uv run python -m src.stopword_analysis` → 언어별 후보 표
    (직접 정한 불용어는 빼고 계산한다. 영어는 NLTK 기본 불용어만 적용된 상태, 한국어는 불용어 없이 계산)
"""

import numpy as np
import pandas as pd

from src.corpus import load_corpus


def stopword_candidates(tokens: pd.Series, labels: pd.Series, min_df_ratio: float = 0.02) -> pd.DataFrame:
    classes = sorted(labels.unique())
    class_size = labels.value_counts()
    rows: dict[str, np.ndarray] = {}
    for toks, lab in zip(tokens, labels):
        k = classes.index(lab)
        for t in set(toks):
            rows.setdefault(t, np.zeros(len(classes)))[k] += 1
    words = list(rows)
    counts = np.array([rows[w] for w in words])  # (단어, 카테고리) 문서 빈도
    df_ratio = counts.sum(axis=1) / len(labels)
    rate = counts / class_size[classes].to_numpy()  # 카테고리 크기 차이를 없앤 출현 비율
    p = rate / rate.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(len(classes))
    out = pd.DataFrame({"word": words, "df_ratio": df_ratio, "evenness": ent})
    out = out[out["df_ratio"] >= min_df_ratio]
    return out.sort_values("df_ratio", ascending=False).reset_index(drop=True)


def main() -> None:
    for lang, tok in (("en", "okt"), ("ko", "okt")):
        corpus = load_corpus(lang, tokenizer=tok, use_stopwords=False)
        cand = stopword_candidates(corpus["tokens"], corpus["label"])
        even = cand[cand["evenness"] >= 0.95]
        print(f"=== {lang}: DF 상위이면서 균등도 >= 0.95 (불용어 후보) ===")
        print(even.head(40).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n=== {lang}: DF 상위이지만 균등도 < 0.95 (주제 신호, 유지) ===")
        print(cand[cand["evenness"] < 0.95].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()


if __name__ == "__main__":
    main()
