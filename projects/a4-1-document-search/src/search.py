"""코사인 유사도 기반 문서 검색 (A4-1).

    1. 말뭉치 전체로 TF-IDF를 학습한다 (어휘 사전, IDF).
    2. 검색어를 문서와 **똑같은 전처리**로 토큰화하고, 말뭉치의 IDF로 벡터화한다.
       어휘 사전에 없는 단어는 버린다 (말뭉치에 없는 단어는 어느 문서와도 겹치지 않는다).
    3. 코사인 유사도 = (q · d) / (||q|| × ||d||) 를 문서 행렬 전체에 한 번에 계산한다.
       내적은 행렬-벡터 곱 `D @ q`, 노름은 `sqrt(sum(x²))`. 0 벡터(남은 단어가 없는 검색어)는 점수 0.
    4. 점수 상위 k개를 (점수, 문서 ID, 스니펫)으로 돌려준다.

함수 사용:
    from src.search import search
    search("space shuttle launch", topk=5)            # -> [(0.41, 'en-01234', '...'), ...]
    search("삼성전자 반도체 실적", lang="ko")
"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.corpus import MIN_DF, load_corpus
from src.preprocess import get_preprocessor
from src.tfidf import NumpyTfidf

SNIPPET_LEN = 90


def cosine_similarity(doc_matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """문서 행렬 (n, V)와 검색어 벡터 (V,)의 코사인 유사도 (n,)."""
    dots = doc_matrix @ query_vec
    doc_norms = np.sqrt(np.einsum("ij,ij->i", doc_matrix, doc_matrix))
    q_norm = np.sqrt(query_vec @ query_vec)
    denom = doc_norms * q_norm
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """점수 상위 k개 인덱스 (내림차순). 전체 정렬 O(n log n) 대신 argpartition O(n) 후 k개만 정렬."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def make_snippet(text: str, query_tokens: list[str], width: int = SNIPPET_LEN) -> str:
    """검색어 단어가 처음 나오는 곳을 중심으로 자른 한 줄 요약. 못 찾으면 앞부분."""
    flat = " ".join(text.split())
    lower = flat.lower()
    # 단어 시작 위치에서만 찾는다: 'art'가 'start' 안에서 걸리지 않게 하고, 'launch'는 'launched'에서 찾는다.
    hits = [m.start() for t in query_tokens if (m := re.search(rf"(?<!\w){re.escape(t)}", lower))]
    start = max(0, min(hits) - width // 3) if hits else 0
    piece = flat[start : start + width]
    return ("…" if start > 0 else "") + piece + ("…" if start + width < len(flat) else "")


class SearchEngine:
    def __init__(self, corpus: pd.DataFrame, preprocessor, vectorizer: NumpyTfidf):
        self.corpus = corpus
        self.prep = preprocessor
        self.vec = vectorizer
        self.matrix = vectorizer.fit_transform(corpus["tokens"].tolist())

    @classmethod
    def build(cls, lang: str = "en", tokenizer: str = "okt", **tfidf_kwargs) -> "SearchEngine":
        corpus = load_corpus(lang, tokenizer=tokenizer)
        vec = NumpyTfidf(min_df=MIN_DF[lang], **tfidf_kwargs)
        return cls(corpus, get_preprocessor(lang, tokenizer=tokenizer), vec)

    def query_vector(self, query: str) -> tuple[np.ndarray, list[str]]:
        tokens = self.prep(query)
        return self.vec.transform([tokens])[0], tokens

    def scores(self, query: str) -> np.ndarray:
        q, _ = self.query_vector(query)
        return cosine_similarity(self.matrix, q)

    def search(self, query: str, topk: int = 5) -> list[tuple[float, str, str]]:
        if topk < 1:
            raise ValueError(f"topk는 1 이상이어야 합니다: {topk}")
        q, tokens = self.query_vector(query)
        scores = cosine_similarity(self.matrix, q)
        if not q.any():
            return []
        results = []
        for i in top_k(scores, topk):
            if scores[i] <= 0:
                break
            row = self.corpus.iloc[i]
            results.append((float(scores[i]), row["doc_id"], make_snippet(row["text"], tokens)))
        return results

    def explain(self, query: str, rank: int = 1) -> str:
        """rank번째 결과 문서와 검색어의 코사인 유사도를 숫자로 풀어 쓴다."""
        q, tokens = self.query_vector(query)
        scores = cosine_similarity(self.matrix, q)
        i = top_k(scores, rank)[rank - 1]
        d = self.matrix[i]
        terms = self.vec.terms_
        known = [t for t in tokens if t in self.vec.vocabulary_]
        unknown = [t for t in tokens if t not in self.vec.vocabulary_]
        q_idx = np.flatnonzero(q)
        dot, qn, dn = float(q @ d), float(np.sqrt(q @ q)), float(np.sqrt(d @ d))
        lines = [
            "=== 코사인 유사도 계산 예시 ===",
            f'Query: "{query}"',
            f"전처리 후 토큰: {tokens}" + (f"  (어휘 사전에 없어 버린 단어: {unknown})" if unknown else ""),
            f"대상 문서: {rank}위 {self.corpus.iloc[i]['doc_id']} [{self.corpus.iloc[i]['label']}]",
            f"벡터 차원: {len(q):,} (검색어에서 0이 아닌 차원 {len(q_idx)}개, 문서에서 {np.count_nonzero(d)}개)",
            "",
            f"{'단어':<14}{'IDF':>8}{'q 가중치':>10}{'d 가중치':>10}{'q×d':>10}",
        ]
        for j in q_idx:
            lines.append(f"{terms[j]:<14}{self.vec.idf_[j]:>8.3f}{q[j]:>10.4f}{d[j]:>10.4f}{q[j] * d[j]:>10.4f}")
        # 정규화하기 전의 TF×IDF 벡터로 같은 계산을 하면 노름이 1이 아니어도 결과가 같다.
        raw = self.vec.term_frequency(self.vec.count_matrix([tokens, self.corpus.iloc[i]["tokens"]])) * self.vec.idf_
        rq, rd = raw
        rdot, rqn, rdn = float(rq @ rd), float(np.sqrt(rq @ rq)), float(np.sqrt(rd @ rd))
        lines += [
            "",
            "코사인 유사도 = (Query · 문서) / (||Query|| × ||문서||)",
            f"  L2 정규화된 벡터: = {dot:.4f} / ({qn:.4f} × {dn:.4f}) = {dot / (qn * dn) if qn * dn else 0.0:.4f}",
            f"  정규화 전 TF×IDF : = {rdot:.4f} / ({rqn:.4f} × {rdn:.4f}) = {rdot / (rqn * rdn) if rqn * rdn else 0.0:.4f}",
            "정규화는 벡터 길이(문서 길이)를 1로 맞출 뿐 방향을 바꾸지 않으므로 두 값이 같다.",
            "공통 단어가 아닌 차원은 q 또는 d 쪽이 0이라 내적에 기여하지 않는다.",
        ]
        return "\n".join(lines) if known else "검색어에서 어휘 사전에 있는 단어가 없어 유사도가 모두 0입니다."


@lru_cache(maxsize=4)
def get_engine(lang: str = "en", tokenizer: str = "okt") -> SearchEngine:
    return SearchEngine.build(lang, tokenizer)


def search(query: str, topk: int = 5, lang: str = "en", tokenizer: str = "okt") -> list[tuple[float, str, str]]:
    """검색 함수 (안 B): search(query) -> List[(score, doc_id, text_snippet)]. 처음 호출 때 색인을 만든다."""
    return get_engine(lang, tokenizer).search(query, topk)


def format_results(query: str, results: list[tuple[float, str, str]], labels: dict[str, str] | None = None) -> str:
    lines = ["=== 검색 결과 ===", f'Query: "{query}"', "", f"Top {len(results)} 결과:"]
    if not results:
        lines.append("  (일치하는 문서 없음: 검색어의 단어가 모두 불용어이거나 어휘 사전에 없습니다)")
    for rank, (score, doc_id, snippet) in enumerate(results, 1):
        tag = f" [{labels[doc_id]}]" if labels else ""
        lines.append(f'{rank}. [{score:.3f}] {doc_id}{tag} "{snippet}"')
    return "\n".join(lines)
