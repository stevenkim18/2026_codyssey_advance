"""코사인 유사도와 검색 엔진 (작은 가짜 말뭉치)."""

import numpy as np
import pandas as pd
from src.preprocess import EnglishPreprocessor
from src.search import SearchEngine, cosine_similarity, make_snippet, top_k
from src.tfidf import NumpyTfidf


def test_cosine_matches_definition():
    d = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    q = np.array([2.0, 1.0, 0.0])
    s = cosine_similarity(d, q)
    np.testing.assert_allclose(s[0], (1 * 2 + 2 * 1) / (np.sqrt(5) * np.sqrt(5)))
    assert s[1] == 0 and s[2] == 0  # 직교, 0 벡터 문서


def test_cosine_zero_query():
    assert not cosine_similarity(np.eye(3), np.zeros(3)).any()


def test_top_k_sorted_desc():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert list(top_k(scores, 3)) == [1, 3, 2]
    assert len(top_k(scores, 10)) == 4


def test_snippet_centers_on_query_word():
    text = "x " * 100 + "the shuttle launch was delayed"
    assert "shuttle" in make_snippet(text, ["shuttle"], width=40)


def _engine():
    texts = [
        "space shuttle launch orbit",
        "shuttle launch delayed by weather",
        "hockey goalie stanley cup playoff",
        "baseball pitcher home run",
        "graphics image format jpeg",
    ]
    prep = EnglishPreprocessor()
    corpus = pd.DataFrame({"doc_id": [f"d{i}" for i in range(5)], "text": texts, "label": list("ssHBG")})
    corpus["tokens"] = corpus["text"].map(prep)
    return SearchEngine(corpus, prep, NumpyTfidf())


def test_search_returns_score_id_snippet():
    results = _engine().search("shuttle launch", topk=5)
    assert [r[1] for r in results][:2] in (["d0", "d1"], ["d1", "d0"])
    assert all(isinstance(s, float) and isinstance(i, str) and isinstance(t, str) for s, i, t in results)
    scores = [r[0] for r in results]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)


def test_search_unknown_or_stopword_query_is_empty():
    eng = _engine()
    assert eng.search("the and of") == []
    assert eng.search("quantum") == []


def test_search_rejects_non_positive_topk():
    import pytest

    with pytest.raises(ValueError):
        _engine().search("shuttle", topk=0)


def test_snippet_matches_word_start_only():
    text = "x " * 60 + "He will start the engine. " + "y " * 60 + "Later, art was discussed."
    snippet = make_snippet(text, ["art"], width=40)
    assert "art was" in snippet and "start" not in snippet
    assert "launched" in make_snippet("z " * 60 + "it launched today", ["launch"], width=30)
