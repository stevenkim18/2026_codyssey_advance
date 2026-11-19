"""직접 구현한 TF-IDF를 scikit-learn TfidfVectorizer와 비교 검증한다 (허용 오차 1e-6).

같은 결과를 보려면 두 구현이 **같은 토큰**과 **같은 수식**을 써야 한다.
    - 토큰: sklearn에 `analyzer=lambda doc: doc`를 줘서 우리 전처리가 만든 토큰 리스트를 그대로 쓰게 한다.
      (sklearn 자체 소문자 변환·토큰 패턴·불용어는 이때 전혀 적용되지 않는다)
    - 수식: smooth_idf, sublinear_tf, norm, min_df를 직접 구현의 설정과 똑같이 맞춘다.
      직접 구현의 tf="log"가 sklearn의 sublinear_tf=True와 같다.
    - 열 순서: 두 구현 모두 어휘를 정렬하지만, 단어 → 열 번호 사전으로 sklearn 열을 다시 맞춘 뒤 비교한다.

실행: `uv run python -m src.verify --lang en` (또는 --lang ko --tokenizer okt)
"""

import argparse
import platform

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from src.corpus import MIN_DF, load_corpus, output_dir
from src.tfidf import NumpyTfidf

TOLERANCE = 1e-6

# (직접 구현 설정, sklearn 설정) 쌍. 첫 번째가 이 프로젝트 기본값이다.
CONFIGS = [
    ({"tf": "raw", "smooth_idf": True, "norm": "l2"}, {"sublinear_tf": False, "smooth_idf": True, "norm": "l2"}),
    ({"tf": "log", "smooth_idf": True, "norm": "l2"}, {"sublinear_tf": True, "smooth_idf": True, "norm": "l2"}),
    ({"tf": "raw", "smooth_idf": False, "norm": "l2"}, {"sublinear_tf": False, "smooth_idf": False, "norm": "l2"}),
    ({"tf": "raw", "smooth_idf": True, "norm": None}, {"sublinear_tf": False, "smooth_idf": True, "norm": None}),
]


def _identity(doc):
    return doc


def compare(docs: list[list[str]], ours_cfg: dict, sk_cfg: dict, min_df: int) -> dict:
    ours = NumpyTfidf(min_df=min_df, **ours_cfg)
    x_ours = ours.fit_transform(docs)

    sk = TfidfVectorizer(analyzer=_identity, lowercase=False, min_df=min_df, dtype=np.float64, **sk_cfg)
    x_sk_sparse = sk.fit_transform(docs)

    sk_terms = sk.get_feature_names_out()
    only_sk = set(sk_terms) - set(ours.vocabulary_)
    only_ours = set(ours.vocabulary_) - set(sk_terms)
    if only_sk or only_ours:  # 어휘가 다르면 행렬을 비교할 수 없으므로 바로 FAIL
        return {
            "shape": x_ours.shape,
            "same_vocab": False,
            "vocab_diff": f"sklearn에만 {len(only_sk)}개, 직접 구현에만 {len(only_ours)}개",
            "idf_max_err": float("nan"),
            "max_err": float("nan"),
            "mean_err": float("nan"),
            "nnz_equal": False,
            "pass": False,
        }
    same_vocab = True
    order = np.array([ours.vocabulary_[t] for t in sk_terms])  # sklearn 열 j → 우리 열 order[j]
    x_sk = np.zeros_like(x_ours)
    x_sk[:, order] = x_sk_sparse.toarray()
    idf_sk = np.empty_like(ours.idf_)
    idf_sk[order] = sk.idf_

    diff = np.abs(x_ours - x_sk)
    return {
        "shape": x_ours.shape,
        "same_vocab": same_vocab,
        "idf_max_err": float(np.abs(ours.idf_ - idf_sk).max()),
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
        "nnz_equal": bool(((x_ours != 0) == (x_sk != 0)).all()),
        "pass": same_vocab and float(diff.max()) <= TOLERANCE,
    }


def run(lang: str, tokenizer: str = "okt") -> str:
    corpus = load_corpus(lang, tokenizer=tokenizer)
    docs = corpus["tokens"].tolist()
    min_df = MIN_DF[lang]
    lines = [
        "=== TF-IDF 구현 검증 결과 ===",
        f"데이터: {lang}{'' if lang == 'en' else f' ({tokenizer})'}, 문서 {len(docs):,}건",
        f"환경: Python {platform.python_version()}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}",
        (
            "공통 sklearn 설정: TfidfVectorizer(analyzer=<전처리된 토큰 그대로>, lowercase=False, "
            f"min_df={min_df}, dtype=float64)"
        ),
        "",
    ]
    for i, (ours_cfg, sk_cfg) in enumerate(CONFIGS):
        r = compare(docs, ours_cfg, sk_cfg, min_df)
        tag = "기본 설정" if i == 0 else "추가 설정"
        sk_desc = ", ".join(f"{k}={v!r}" for k, v in sk_cfg.items())
        ours_desc = ", ".join(f"{k}={v!r}" for k, v in ours_cfg.items())
        lines += [
            f"[검증 {i + 1}: {tag}] 직접 구현 vs Scikit-learn",
            f"  - 직접 구현 설정: NumpyTfidf({ours_desc}, min_df={min_df})",
            f"  - sklearn 설정 : {sk_desc}",
            f"  - TF-IDF 행렬 shape: {r['shape']}, 어휘 일치: {r['same_vocab']}, 0이 아닌 위치 일치: {r['nnz_equal']}"
            + (f" ({r['vocab_diff']})" if "vocab_diff" in r else ""),
            f"  - IDF 최대 오차: {r['idf_max_err']:.2e}",
            f"  - 최대 오차: {r['max_err']:.2e}",
            f"  - 평균 오차: {r['mean_err']:.2e}",
            f"  - 결과: {'PASS' if r['pass'] else 'FAIL'} (허용 오차 {TOLERANCE:g} 이내)",
            "",
        ]
    text = "\n".join(lines)
    path = output_dir(lang, tokenizer) / "verify_log.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return text + f"→ 저장: {path.relative_to(path.parents[2])}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--tokenizer", choices=["okt", "whitespace"], default="okt")
    args = ap.parse_args()
    print(run(args.lang, args.tokenizer))
