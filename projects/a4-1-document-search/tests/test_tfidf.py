"""직접 구현 TF-IDF: 손으로 계산한 값과 scikit-learn 결과와 비교."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from src.tfidf import NumpyTfidf

DOCS = [["a", "b", "a"], ["b", "c"], ["a"]]


def test_vocabulary_is_sorted_and_df():
    v = NumpyTfidf().fit(DOCS)
    assert list(v.terms_) == ["a", "b", "c"]
    np.testing.assert_array_equal(v.df_, [2, 2, 1])


def test_counts_and_smooth_idf_by_hand():
    v = NumpyTfidf(norm=None).fit(DOCS)
    np.testing.assert_array_equal(v.count_matrix(DOCS), [[2, 1, 0], [0, 1, 1], [1, 0, 0]])
    n = 3
    expected_idf = [np.log((1 + n) / (1 + 2)) + 1, np.log((1 + n) / (1 + 2)) + 1, np.log((1 + n) / (1 + 1)) + 1]
    np.testing.assert_allclose(v.idf_, expected_idf)
    x = v.transform(DOCS)
    np.testing.assert_allclose(x[0], [2 * expected_idf[0], expected_idf[1], 0])


def test_l2_rows_have_unit_norm():
    x = NumpyTfidf().fit_transform(DOCS)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0)


def test_unsmoothed_idf():
    v = NumpyTfidf(smooth_idf=False).fit(DOCS)
    np.testing.assert_allclose(v.idf_, [np.log(3 / 2) + 1, np.log(3 / 2) + 1, np.log(3) + 1])


def test_tf_variants():
    counts = np.array([[2.0, 1.0, 0.0]])
    v = NumpyTfidf(tf="log")
    np.testing.assert_allclose(v.term_frequency(counts), [[1 + np.log(2), 1, 0]])
    v = NumpyTfidf(tf="double")
    np.testing.assert_allclose(v.term_frequency(counts), [[1.0, 0.75, 0]])
    v = NumpyTfidf(tf="binary")
    np.testing.assert_allclose(v.term_frequency(counts), [[1, 1, 0]])


def test_unknown_words_ignored_and_empty_doc_is_zero():
    v = NumpyTfidf().fit(DOCS)
    x = v.transform([["zzz"], ["a", "zzz"]])
    assert not x[0].any()
    np.testing.assert_allclose(x[1], [1, 0, 0])


def test_min_df_filters_rare_terms():
    v = NumpyTfidf(min_df=2).fit(DOCS)
    assert list(v.terms_) == ["a", "b"]


@pytest.mark.parametrize(
    ("ours", "sk"),
    [
        ({"tf": "raw", "smooth_idf": True, "norm": "l2"}, {"sublinear_tf": False, "smooth_idf": True, "norm": "l2"}),
        ({"tf": "log", "smooth_idf": True, "norm": "l2"}, {"sublinear_tf": True, "smooth_idf": True, "norm": "l2"}),
        ({"tf": "raw", "smooth_idf": False, "norm": None}, {"sublinear_tf": False, "smooth_idf": False, "norm": None}),
    ],
)
def test_matches_sklearn_on_random_corpus(ours, sk):
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(60)]
    docs = [list(rng.choice(words, size=rng.integers(1, 40))) for _ in range(200)]
    x_ours = NumpyTfidf(min_df=2, **ours).fit_transform(docs)
    x_sk = TfidfVectorizer(analyzer=lambda d: d, lowercase=False, min_df=2, **sk).fit_transform(docs).toarray()
    assert np.abs(x_ours - x_sk).max() < 1e-6


def test_verify_reports_fail_when_vocab_differs(monkeypatch):
    from src import verify

    class DropFirstTerm(NumpyTfidf):  # 어휘 하나를 빠뜨리는 잘못된 구현
        def fit(self, docs):
            super().fit(docs)
            terms = list(self.terms_[1:])
            self.vocabulary_ = {t: i for i, t in enumerate(terms)}
            self.terms_ = np.array(terms, dtype=object)
            self.df_, self.idf_ = self.df_[1:], self.idf_[1:]
            return self

    monkeypatch.setattr(verify, "NumpyTfidf", DropFirstTerm)
    r = verify.compare(DOCS, {"tf": "raw"}, {"sublinear_tf": False}, min_df=1)
    assert r["pass"] is False and r["same_vocab"] is False
