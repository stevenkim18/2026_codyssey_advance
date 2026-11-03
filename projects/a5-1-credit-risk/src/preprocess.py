"""데이터 로딩, Train/Test 분할, 전처리 파이프라인 (A5-1).

구성
    1. 데이터 로딩: `data/finance_data.csv` 를 읽는다. 없으면 `src.data_gen` 실행을 안내한다.
    2. 연령대 변환: `age` 를 `age_group`(20s~60s) 으로 바꿔 범주형 변수를 만든다.
       경계값이 고정이라 데이터에서 배우는 값이 없고, 그래서 분할 전에 해도 누수가 아니다.
    3. 분할: 8:2, 연체 여부 기준 stratified, random_state=42. 회귀와 분류가 같은 분할을 쓴다.
    4. 전처리: `ColumnTransformer` 하나로 수치형은 대치 → 스케일링, 범주형은 대치 → 원-핫 인코딩을 한다.

데이터 누수를 막는 장치
    - 입력 feature 목록(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`)에는 두 타깃(`credit_score`,
      `is_overdue`)이 없다. 특히 연체 여부는 `credit_score` 하위 15% 에서 만들어지므로,
      `credit_score` 를 분류 feature 로 쓰면 정답을 미리 알려 주는 셈이다.
    - 이 모듈은 전처리기를 fit 하지 않은 채로 돌려준다. fit 은 호출하는 쪽에서 Pipeline 안에
      넣어 Train 데이터로만 하고, 교차 검증에서는 Pipeline 이 폴드마다 다시 fit 한다.

실행: `uv run python -m src.preprocess` (프로젝트 폴더에서) 하면 분할 현황과 전처리 검증 결과를 출력한다.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_gen import DATA_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # 교차 검증 fold 수

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_REG = "credit_score"  # 회귀 타깃 (0~1000)
TARGET_CLF = "is_overdue"  # 분류 타깃 (1 = 연체)

NUMERIC_FEATURES = [
    "annual_income",
    "spending_score",
    "debt_ratio",
    "credit_card_count",
    "overdue_count_6m",
]
CATEGORICAL_FEATURES = ["age_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 입력 변수 6개

AGE_BINS = [20, 30, 40, 50, 60, 70]  # [20, 30) 은 20s ... [60, 70) 은 60s
AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]


class Split(NamedTuple):
    """Train/Test 분할 결과. 회귀와 분류가 같은 행을 쓰도록 타깃 둘을 함께 나눈다."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_clf_train: pd.Series
    y_clf_test: pd.Series
    y_reg_train: pd.Series
    y_reg_test: pd.Series


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """생성된 CSV 를 읽는다."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `uv run python -m src.data_gen` 으로 데이터를 생성하세요."
        )
    return pd.read_csv(path)


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """`age` 를 연령대 `age_group` 으로 바꾼다 (원본 CSV 는 건드리지 않고 복사본을 돌려준다)."""
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return out.drop(columns="age")


def split_data(df: pd.DataFrame) -> Split:
    """연령대 변환 후 8:2 로 나눈다. 연체 비율(12%)이 Train/Test 에서 같도록 stratify 한다."""
    df = add_age_group(df)
    X = df[FEATURES]
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X,
        df[TARGET_CLF],
        df[TARGET_REG],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_CLF],
    )
    return Split(X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test)


def make_cv() -> StratifiedKFold:
    """5-fold 교차 검증. 회귀에서도 연체 여부로 층화해 fold 마다 분포를 맞춘다."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def build_preprocessor(numeric_features: list[str] | None = None) -> ColumnTransformer:
    """수치형: 중앙값 대치 → 표준화, 범주형: 최빈값 대치 → 원-핫 인코딩.

    fit 하지 않은 상태로 돌려주므로 반드시 `Pipeline` 에 넣어 Train 데이터로만 fit 한다.
    이 데이터에는 결측치가 없어서 대치기는 실제로는 값을 바꾸지 않지만, 결측이 생기는 새 데이터가
    들어와도 Train 의 중앙값·최빈값으로 채우도록 파이프라인에 포함했다.

    numeric_features 를 생략하면 `NUMERIC_FEATURES` 를 쓴다. 다른 목록은 데이터 누수 실험
    (`credit_score` 를 일부러 넣어 보는 경우)에서만 넘긴다.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return pre.set_output(transform="pandas")


def clean_feature_names(names) -> list[str]:
    """`num__annual_income` → `annual_income`, `cat__age_group_20s` → `age_group_20s`."""
    return [n.split("__", 1)[1] for n in names]


def main() -> None:
    df = load_data()
    print(f"데이터: {df.shape[0]:,}행 x {df.shape[1]}열, 결측치 {int(df.isna().sum().sum())}개")
    print(f"연체 비율: {df[TARGET_CLF].mean():.4f}\n")

    split = split_data(df)
    print("[분할] random_state=42, stratify=is_overdue")
    for name, X, y in (
        ("Train", split.X_train, split.y_clf_train),
        ("Test", split.X_test, split.y_clf_test),
    ):
        print(f"  {name:5s} {len(X):>5,}행  연체 {int(y.sum()):>4,}건 ({y.mean():.4f})")
    assert TARGET_REG not in FEATURES and TARGET_CLF not in FEATURES
    print(f"  입력 변수 {len(FEATURES)}개: {FEATURES}  (타깃은 포함하지 않음)\n")

    # Train 으로만 fit 했는지 확인: Train 은 평균 0 / 표준편차 1, Test 는 약간 어긋난다.
    pre = build_preprocessor().fit(split.X_train)
    train_t = pre.transform(split.X_train)
    test_t = pre.transform(split.X_test)
    cols = clean_feature_names(train_t.columns)
    train_t.columns = test_t.columns = cols
    print("[전처리 검증] 수치형 변환 후 평균 (Train 에서 fit)")
    check = pd.DataFrame(
        {
            "train_mean": train_t[NUMERIC_FEATURES].mean(),
            "test_mean": test_t[NUMERIC_FEATURES].mean(),
        }
    )
    print(check.round(4).to_string())
    print(f"\n변환 후 열 {train_t.shape[1]}개: {cols}")
    assert np.allclose(check["train_mean"], 0, atol=1e-9)


if __name__ == "__main__":
    main()
