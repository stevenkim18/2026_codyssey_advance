"""이커머스 거래 데이터 분석 파이프라인 (A1-1).

H&M 샘플(거래 + 상품 + 고객)을 불러와 전처리, 피처 엔지니어링, 통계 요약,
RFM 고객 세분화를 수행하는 `DataAnalyzer` 클래스를 제공한다.

사용 예:
    analyzer = DataAnalyzer()
    analyzer.load_data()
    analyzer.add_text_features("prod_name")
    images, ids = analyzer.load_image_arrays()
    analyzer.add_image_features(images, ids)
    analyzer.handle_missing_values("img_mean", "group_mean", "product_group_name", unit_col="article_id")
    outliers = analyzer.detect_outliers("amount", threshold=1.5)
    rfm = analyzer.calculate_rfm("customer_id", "t_dat", "amount")
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

MISSING_STRATEGIES = ("mean", "median", "group_mean", "group_median", "drop")


class DataAnalyzer:
    """거래 데이터를 하나의 DataFrame(`self.df`)에 담고 단계별로 가공한다.

    Attributes:
        data_dir: `sample/`, `processed/` 하위 폴더를 가진 데이터 폴더
        df: 거래 단위 분석 테이블 (거래 + 상품 + 고객 + 파생 변수)
        raw_df: `load_data()` 직후의 원본 복사본 (전처리 전후 비교용)
        rfm: `calculate_rfm()` 결과 (고객 단위)
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.df: pd.DataFrame | None = None
        self.raw_df: pd.DataFrame | None = None
        self.rfm: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # 1. 로드
    # ------------------------------------------------------------------
    def load_data(self, price_scale: float = 1000.0) -> pd.DataFrame:
        """거래, 상품, 고객 샘플을 읽어 거래 단위 테이블로 합친다.

        원본 `price`는 정규화된 상대 금액(중앙값 약 0.025)이라 읽기 어렵다.
        `price_scale`을 곱한 `amount` 컬럼을 추가한다. 수량 컬럼이 없으므로
        한 거래 행은 1개 구매이며 `amount = price * price_scale`이다.
        """
        sample = self.data_dir / "sample"
        transactions = pd.read_csv(
            sample / "transactions_sample.csv",
            dtype={"article_id": str},
            parse_dates=["t_dat"],
        )
        articles = pd.read_csv(
            sample / "articles_sample.csv", dtype={"article_id": str}
        )
        customers = pd.read_csv(sample / "customers_sample.csv")

        df = transactions.merge(articles, on="article_id", how="left")
        df = df.merge(customers, on="customer_id", how="left")
        df["amount"] = df["price"] * price_scale

        self.df = df
        self.raw_df = df.copy()
        return self.df

    def load_image_arrays(
        self, array_path: str | Path | None = None, ids_path: str | Path | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """`build_image_array.py`가 만든 이미지 배열과 article_id 목록을 읽는다."""
        processed = self.data_dir / "processed"
        array_path = Path(array_path or processed / "image_arrays.npy")
        ids_path = Path(ids_path or processed / "image_article_ids.csv")
        images = np.load(array_path)
        ids = pd.read_csv(ids_path, dtype={"article_id": str})["article_id"].tolist()
        if len(ids) != len(images):
            raise ValueError(
                f"이미지 {len(images)}장과 id {len(ids)}개의 수가 다릅니다"
            )
        return images, ids

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("먼저 load_data()를 호출하세요")
        return self.df

    # ------------------------------------------------------------------
    # 2. 피처 엔지니어링
    # ------------------------------------------------------------------
    def add_text_features(self, column: str = "prod_name") -> pd.DataFrame:
        """텍스트 컬럼에서 단어 수와 문자열 길이 파생 변수를 추가한다."""
        df = self._require_df()
        text = df[column].astype("string")
        df[f"{column}_word_count"] = text.str.split().str.len()
        df[f"{column}_len"] = text.str.len()
        return df

    def add_image_features(
        self, images: np.ndarray, article_ids: Sequence[str]
    ) -> pd.DataFrame:
        """이미지 배열 `(N, H, W, C)`의 평균/표준편차를 반복문 없이 계산해 추가한다.

        이미지별로 H, W, C 세 축을 한 번에 집계하므로 for 문이 필요 없다.
        이미지가 없는 상품은 `img_mean`, `img_std`가 결측이 되고
        `has_image`는 False가 된다.
        """
        df = self._require_df()
        if images.ndim != 4:
            raise ValueError("images는 (N, H, W, C) 4차원 배열이어야 합니다")
        features = pd.DataFrame(
            {
                "article_id": list(article_ids),
                "img_mean": images.mean(axis=(1, 2, 3)),
                "img_std": images.std(axis=(1, 2, 3)),
            }
        )
        df = df.drop(columns=["img_mean", "img_std", "has_image"], errors="ignore")
        df = df.merge(features, on="article_id", how="left")
        df["has_image"] = df["img_mean"].notna()
        self.df = df
        return self.df

    # ------------------------------------------------------------------
    # 3. 결측치
    # ------------------------------------------------------------------
    def missing_report(self) -> pd.DataFrame:
        """컬럼별 결측 개수와 비율(%)을 결측이 많은 순으로 반환한다."""
        df = self._require_df()
        report = pd.DataFrame(
            {"missing": df.isna().sum(), "missing_pct": df.isna().mean() * 100}
        )
        return report[report["missing"] > 0].sort_values("missing", ascending=False)

    def handle_missing_values(
        self,
        column: str,
        strategy: str = "group_median",
        group_col: str | None = None,
        unit_col: str | None = None,
    ) -> pd.DataFrame:
        """수치형 컬럼의 결측치를 대치한다.

        Args:
            column: 대치할 컬럼
            strategy: mean, median, group_mean, group_median, drop 중 하나
            group_col: group_* 전략에서 묶을 기준 컬럼
            unit_col: 지정하면 이 컬럼 기준으로 중복을 제거한 뒤 통계를 계산한다.
                거래 행 수가 아니라 상품 수 기준으로 평균을 내고 싶을 때 쓴다
                (예: `unit_col="article_id"`).

        group_* 전략에서 그룹 통계가 없는 행(그룹 값 결측, 그룹 전체가 결측)은
        전체 평균/중앙값으로 대신 채운다.
        """
        df = self._require_df()
        if strategy not in MISSING_STRATEGIES:
            raise ValueError(
                f"알 수 없는 전략: {strategy} (가능: {MISSING_STRATEGIES})"
            )

        if strategy == "drop":
            self.df = df.dropna(subset=[column]).reset_index(drop=True)
            return self.df

        stat = "mean" if strategy.endswith("mean") else "median"
        base = df.drop_duplicates(unit_col) if unit_col else df
        overall = getattr(base[column], stat)()

        if strategy.startswith("group_"):
            if group_col is None:
                raise ValueError(f"{strategy} 전략에는 group_col이 필요합니다")
            group_stat = base.groupby(group_col)[column].agg(stat)
            fill = df[group_col].map(group_stat).fillna(overall)
        else:
            fill = overall

        df[column] = df[column].fillna(fill)
        return df

    # ------------------------------------------------------------------
    # 4. 이상치 (IQR)
    # ------------------------------------------------------------------
    def iqr_bounds(self, column: str, threshold: float = 1.5) -> tuple[float, float]:
        """IQR 방식의 (하한, 상한)을 반환한다.

        하한 = Q1 - threshold * IQR, 상한 = Q3 + threshold * IQR
        """
        series = self._require_df()[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr

    def detect_outliers(self, column: str, threshold: float = 1.5) -> pd.DataFrame:
        """IQR 범위를 벗어난 행을 반환한다."""
        df = self._require_df()
        lower, upper = self.iqr_bounds(column, threshold)
        return df[(df[column] < lower) | (df[column] > upper)]

    def treat_outliers(
        self, column: str, threshold: float = 1.5, method: str = "clip"
    ) -> pd.DataFrame:
        """이상치를 처리한 새 DataFrame을 반환한다 (`self.df`는 바뀌지 않는다).

        method="clip"은 상/하한 값으로 대체하고 행 수를 유지한다.
        method="remove"는 이상치 행을 삭제한다.
        """
        df = self._require_df()
        lower, upper = self.iqr_bounds(column, threshold)
        if method == "clip":
            treated = df.copy()
            treated[column] = treated[column].clip(lower, upper)
            return treated
        if method == "remove":
            is_outlier = (df[column] < lower) | (df[column] > upper)
            return df[~is_outlier].reset_index(drop=True)
        raise ValueError(f"알 수 없는 method: {method} (가능: clip, remove)")

    # ------------------------------------------------------------------
    # 5. 통계
    # ------------------------------------------------------------------
    def summary_statistics(self, columns: Sequence[str]) -> pd.DataFrame:
        """평균, 중앙값, 표준편차, 사분위수, IQR을 컬럼별로 계산한다."""
        df = self._require_df()
        rows = {}
        for col in columns:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            rows[col] = {
                "count": s.size,
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "q1": q1,
                "q3": q3,
                "max": s.max(),
                "iqr": q3 - q1,
            }
        return pd.DataFrame(rows).T

    def correlation_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """변수 쌍별 피어슨 상관계수를 반환한다. `data`로 대상 테이블을 바꿀 수 있다."""
        table = self._require_df() if data is None else data
        rows = [
            {
                "x": x,
                "y": y,
                "n": table[[x, y]].dropna().shape[0],
                "r": table[x].corr(table[y]),
            }
            for x, y in pairs
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. RFM
    # ------------------------------------------------------------------
    @staticmethod
    def _score(series: pd.Series, n_bins: int, higher_is_better: bool) -> pd.Series:
        """값을 1~n_bins 점수로 바꾼다. 같은 값은 항상 같은 점수를 받는다.

        평균 순위의 백분위(0~1)를 n_bins 구간으로 올림해 점수를 만든다.
        `rank(method="first")`는 동점 고객을 임의로 갈라 놓지만 이 방식은
        동점을 한 점수로 묶는다. 대신 구간별 인원은 정확히 같지 않다.
        """
        ranked = series if higher_is_better else -series
        pct = ranked.rank(method="average", pct=True)
        return np.ceil(pct * n_bins).clip(1, n_bins).astype(int)

    @staticmethod
    def assign_segments(rfm: pd.DataFrame) -> pd.Series:
        """R, F, M 점수(1~5)로 세그먼트 이름을 붙인다. 위 조건이 먼저 적용된다."""
        r, f, m = rfm["R"], rfm["F"], rfm["M"]
        conditions = [
            (r >= 4) & (f >= 4) & (m >= 4),  # 최근, 자주, 많이
            (r >= 3) & (f >= 3),  # 꾸준히 구매
            (r >= 4) & (f <= 2),  # 최근에 처음/가끔 구매
            (r <= 2) & (f >= 3),  # 예전엔 자주 샀지만 최근 소식 없음
            (r <= 2),  # 오래전 마지막 구매
        ]
        names = ["VIP", "Loyal", "New", "At Risk", "Churned"]
        return pd.Series(
            np.select(conditions, names, default="Regular"), index=rfm.index
        )

    def calculate_rfm(
        self,
        customer_col: str = "customer_id",
        date_col: str = "t_dat",
        amount_col: str = "amount",
        n_bins: int = 5,
        snapshot_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """고객별 Recency, Frequency, Monetary와 점수, 세그먼트를 계산한다.

        - Recency: 기준일 - 마지막 구매일(일). 작을수록 좋다.
        - Frequency: 구매한 서로 다른 날짜의 수. 같은 날 여러 상품을 사도 1회다.
        - Monetary: 구매 금액 합계.
        기준일(snapshot_date)의 기본값은 마지막 거래일 + 1일이다.
        """
        df = self._require_df()
        snapshot = (
            pd.Timestamp(snapshot_date)
            if snapshot_date is not None
            else df[date_col].max() + pd.Timedelta(days=1)
        )
        grouped = df.groupby(customer_col)
        rfm = pd.DataFrame(
            {
                "recency": (snapshot - grouped[date_col].max()).dt.days,
                "frequency": grouped[date_col].nunique(),
                "monetary": grouped[amount_col].sum(),
            }
        )
        rfm["R"] = self._score(rfm["recency"], n_bins, higher_is_better=False)
        rfm["F"] = self._score(rfm["frequency"], n_bins, higher_is_better=True)
        rfm["M"] = self._score(rfm["monetary"], n_bins, higher_is_better=True)
        rfm["segment"] = self.assign_segments(rfm)

        self.rfm = rfm
        return self.rfm

    def segment_summary(self, rfm: pd.DataFrame | None = None) -> pd.DataFrame:
        """세그먼트별 고객 수, 평균 R/F/M, 매출 비중을 요약한다."""
        rfm = self.rfm if rfm is None else rfm
        if rfm is None:
            raise RuntimeError("먼저 calculate_rfm()을 호출하세요")
        summary = rfm.groupby("segment").agg(
            customers=("monetary", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        summary["customer_pct"] = (
            summary["customers"] / summary["customers"].sum() * 100
        )
        summary["revenue_pct"] = (
            summary["total_monetary"] / summary["total_monetary"].sum() * 100
        )
        return summary.sort_values("avg_monetary", ascending=False)
