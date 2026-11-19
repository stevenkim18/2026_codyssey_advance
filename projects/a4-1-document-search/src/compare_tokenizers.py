"""한국어 토큰화 비교: Okt 형태소 분석 vs 공백 토큰화 (같은 7,000건, 같은 분할, 같은 모델).

공백 토큰화는 조사·어미가 붙은 어절을 그대로 단어로 쓴다. "삼성전자가 / 삼성전자는 / 삼성전자"가
서로 다른 열이 되어 같은 개념의 문서 빈도가 여러 열로 쪼개진다. 그 결과를 숫자로 확인한다.
    - 어휘 크기 (min_df=1), 한 번만 나온 단어 비율
    - Test 토큰 중 Train 어휘에 없는 비율 (OOV): 처음 보는 어절은 분류·검색에 아무 정보도 주지 못한다
    - 분류 정확도·macro F1
    - 같은 검색어로 찾은 문서 수와 상위 결과

실행: `uv run python -m src.compare_tokenizers`
"""

import numpy as np

from src.classify import run_classification, split
from src.corpus import load_corpus
from src.search import SearchEngine, format_results
from src.tfidf import NumpyTfidf

EXAMPLES = [
    "삼성전자가 3분기 사상 최대 실적을 냈다",
    "손흥민 시즌 첫 골 쏜다",
    "北미사일 발사 정부 개성공단 체류인원 축소",
]
QUERIES = ["삼성전자 실적", "대통령 정상회담"]


def corpus_stats(corpus, lang: str = "ko") -> dict:
    train, test = split(corpus)
    full_vocab = NumpyTfidf(min_df=1).fit(corpus["tokens"].tolist())
    train_vocab = set(NumpyTfidf(min_df=1).fit(train["tokens"].tolist()).vocabulary_)
    test_tokens = [t for doc in test["tokens"] for t in doc]
    return {
        "vocab_min_df1": len(full_vocab.vocabulary_),
        "hapax_ratio": float(np.mean(full_vocab.df_ == 1)),
        "tokens_per_doc": float(corpus["tokens"].map(len).mean()),
        "test_oov_ratio": float(np.mean([t not in train_vocab for t in test_tokens])),
    }


def main() -> None:
    from src.preprocess import KoreanPreprocessor

    print("=== 같은 문장, 다른 토큰화 ===")
    for s in EXAMPLES:
        print(f'"{s}"')
        for tok in ("okt", "whitespace"):
            print(f"  {tok:<10}: {KoreanPreprocessor(tok)(s)}")
    print()

    rows = {}
    engines = {}
    for tok in ("okt", "whitespace"):
        corpus = load_corpus("ko", tokenizer=tok)
        stats = corpus_stats(corpus)
        res = run_classification(corpus, "ko", f"ko-{tok}")
        stats.update(
            {
                "docs": len(corpus),
                "vocab_train_min_df2": len(res.vectorizer.vocabulary_),
                "accuracy": res.accuracy,
                "f1_macro": res.f1_macro,
            }
        )
        rows[tok] = stats
        engines[tok] = SearchEngine.build("ko", tok)

    print("=== Okt vs 공백 토큰화 (KLUE-YNAT 7,000건, Train/Test 8:2, LogReg) ===")
    print(f"{'지표':<32}{'okt':>12}{'whitespace':>12}")
    fmt = {
        "docs": ("문서 수 (토큰 2개 이상)", "{:,}"),
        "tokens_per_doc": ("문서당 토큰 수", "{:.2f}"),
        "vocab_min_df1": ("어휘 크기 (min_df=1)", "{:,}"),
        "hapax_ratio": ("한 문서에만 나온 단어 비율", "{:.1%}"),
        "vocab_train_min_df2": ("어휘 크기 (Train, min_df=2)", "{:,}"),
        "test_oov_ratio": ("Test 토큰 중 Train에 없는 비율", "{:.1%}"),
        "accuracy": ("정확도", "{:.3f}"),
        "f1_macro": ("F1 (macro)", "{:.3f}"),
    }
    for key, (label, f) in fmt.items():
        print(f"{label:<32}{f.format(rows['okt'][key]):>12}{f.format(rows['whitespace'][key]):>12}")
    print()

    for q in QUERIES:
        for tok in ("okt", "whitespace"):
            eng = engines[tok]
            n_hit = int((eng.scores(q) > 0).sum())
            labels = dict(zip(eng.corpus["doc_id"], eng.corpus["label"]))
            print(f"--- {tok}: 점수 > 0 인 문서 {n_hit}건 ---")
            print(format_results(q, eng.search(q, 3), labels), "\n")


if __name__ == "__main__":
    main()
