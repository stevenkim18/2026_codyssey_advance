"""NumPy만으로 구현한 TF-IDF (A4-1).

입력은 이미 전처리된 토큰 리스트의 리스트다 (문서 n개). 출력은 (n, V) 밀집 행렬.

    TF(t, d)   = 문서 d 안에서 단어 t가 나온 횟수                     (tf="raw", 기본값)
                 1 + ln(횟수), 횟수가 0이면 0                         (tf="log", sklearn sublinear_tf=True)
                 0.5 + 0.5 * 횟수 / max_t' 횟수(t', d), 횟수가 0이면 0 (tf="double", 보너스)
                 1 if 횟수 > 0 else 0                                (tf="binary", 보너스)
    IDF(t)     = ln((1 + n) / (1 + df(t))) + 1                         (smooth_idf=True, 기본값)
                 ln(n / df(t)) + 1                                     (smooth_idf=False)
    TF-IDF     = TF * IDF, 그 뒤 문서 벡터마다 L2 정규화 (norm="l2")

df(t)는 단어 t가 한 번 이상 나온 문서 수다. IDF 끝의 "+1"은 모든 문서에 나온 단어도
가중치가 0이 되지 않게 하려는 scikit-learn의 정의이고, smooth의 "+1"은 가상의 문서 하나가
모든 단어를 한 번씩 포함한다고 보고 0으로 나누기를 막는다.

어휘 사전은 알파벳(유니코드)순으로 정렬한다. scikit-learn TfidfVectorizer와 열 순서가 같아진다.
"""

import numpy as np

TF_VARIANTS = ("raw", "log", "double", "binary")


class NumpyTfidf:
    def __init__(
        self,
        tf: str = "raw",
        smooth_idf: bool = True,
        norm: str | None = "l2",
        min_df: int = 1,
        max_df: float = 1.0,
        dtype=np.float64,
    ):
        if tf not in TF_VARIANTS:
            raise ValueError(f"tf는 {TF_VARIANTS} 중 하나: {tf}")
        if norm not in ("l2", None):
            raise ValueError(f"norm은 'l2' 또는 None: {norm}")
        self.tf = tf
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.min_df = min_df
        self.max_df = max_df
        self.dtype = dtype

    # ---------- 학습: 어휘 사전, 문서 빈도, IDF ----------
    def fit(self, docs: list[list[str]]) -> "NumpyTfidf":
        n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        max_count = self.max_df * n if isinstance(self.max_df, float) else self.max_df
        terms = sorted(t for t, c in df.items() if self.min_df <= c <= max_count)
        if not terms:
            raise ValueError("min_df/max_df 조건을 만족하는 단어가 없습니다.")

        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        self.terms_ = np.array(terms, dtype=object)
        self.n_docs_ = n
        self.df_ = np.array([df[t] for t in terms], dtype=np.float64)
        if self.smooth_idf:
            self.idf_ = np.log((1.0 + n) / (1.0 + self.df_)) + 1.0
        else:
            self.idf_ = np.log(n / self.df_) + 1.0
        return self

    # ---------- 변환: 횟수 행렬 → TF → TF-IDF → 정규화 ----------
    def count_matrix(self, docs: list[list[str]]) -> np.ndarray:
        """(n, V) 단어 횟수 행렬. 어휘 사전에 없는 단어는 버린다."""
        vocab = self.vocabulary_
        rows, cols = [], []
        for i, doc in enumerate(docs):
            for t in doc:
                j = vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        counts = np.zeros((len(docs), len(vocab)), dtype=self.dtype)
        np.add.at(counts, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)), 1.0)
        return counts

    def term_frequency(self, counts: np.ndarray) -> np.ndarray:
        if self.tf == "raw":
            return counts
        if self.tf == "binary":
            return (counts > 0).astype(self.dtype)
        if self.tf == "log":
            out = np.zeros_like(counts)
            np.log(counts, out=out, where=counts > 0)
            return np.where(counts > 0, out + 1.0, 0.0)
        # double normalization (K = 0.5): 긴 문서가 횟수만으로 유리해지는 것을 막는다
        max_per_doc = counts.max(axis=1, keepdims=True)
        safe = np.where(max_per_doc > 0, max_per_doc, 1.0)
        return np.where(counts > 0, 0.5 + 0.5 * counts / safe, 0.0)

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        x = self.term_frequency(self.count_matrix(docs))
        x *= self.idf_  # (n, V) * (V,) 브로드캐스팅. 메모리를 아끼려고 제자리 연산
        if self.norm == "l2":
            norms = np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]
            x /= np.where(norms > 0, norms, 1.0)  # 단어가 하나도 없는 문서는 0 벡터로 둔다
        return x

    def fit_transform(self, docs: list[list[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)
