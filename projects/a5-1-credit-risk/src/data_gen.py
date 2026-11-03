"""가상 금융 데이터 생성 (A5-1): 신용 점수(회귀 타깃)와 연체 여부(분류 타깃)를 함께 만든다.

과제가 제공한 데이터 생성 코드를 그대로 옮겼다. 달라진 점은 세 가지뿐이고, 생성되는 데이터는
원본 코드와 같다.
    - 쓰지 않는 `import random` 을 뺐다.
    - 함수(`generate_data`)로 감싸 다른 모듈과 테스트에서 불러 쓸 수 있게 했다.
    - 저장 경로를 `data/finance_data.csv` 로 정했다 (`*.csv` 는 .gitignore 로 제외된다).

데이터에는 의도적인 노이즈와 불균형이 들어 있으므로 생성 후에 값을 고치지 않는다.
난수 호출 순서가 곧 데이터이므로, 난수를 쓰는 줄의 순서를 바꾸면 안 된다.

실행: `uv run python -m src.data_gen` (프로젝트 폴더에서)
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 10000  # 샘플 개수
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "finance_data.csv"


def generate_data() -> pd.DataFrame:
    """과제 원본과 같은 순서로 난수를 뽑아 10,000건의 고객 데이터를 만든다."""
    np.random.seed(RANDOM_STATE)

    # 1. Feature 생성 (가상의 고객 정보)
    data = {
        "age": np.random.randint(20, 70, N_SAMPLES),  # 나이
        "annual_income": np.random.normal(5000, 2000, N_SAMPLES).round(0),  # 연 소득 (만원)
        "spending_score": np.random.randint(1, 100, N_SAMPLES),  # 소비 점수
        "debt_ratio": np.random.uniform(0, 1, N_SAMPLES).round(2),  # 소득 대비 부채 비율
        "credit_card_count": np.random.randint(1, 10, N_SAMPLES),  # 보유 신용카드 수
        "overdue_count_6m": np.random.poisson(0.5, N_SAMPLES),  # 최근 6개월 연체 횟수
    }
    df = pd.DataFrame(data)

    # 소득이 너무 낮은 경우 1500 으로 올린다 (원본 코드의 노이즈 처리: 주석은 0 이라 했지만 코드는 1500)
    df["annual_income"] = df["annual_income"].apply(lambda x: max(x, 1500))

    # 2. Target 1: 신용 점수 (회귀) - 소득이 높고 연체가 적을수록 높고, 노이즈가 더해진다
    df["credit_score"] = (
        300
        + (df["annual_income"] / 100) * 3
        - (df["overdue_count_6m"] * 50)
        - (df["debt_ratio"] * 100)
        + np.random.normal(0, 30, N_SAMPLES)
    )
    df["credit_score"] = df["credit_score"].clip(0, 1000).round(0)  # 0 ~ 1000점으로 제한

    # 3. Target 2: 연체 여부 (분류) - 신용 점수 하위 15% 중 80% 가 연체 (불균형 데이터)
    threshold = df["credit_score"].quantile(0.15)
    df["is_overdue"] = np.where(
        (df["credit_score"] < threshold) & (np.random.rand(N_SAMPLES) > 0.2),
        1,  # 연체 (Positive)
        0,  # 정상 (Negative)
    )

    # 4. 데이터 셔플
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    df = generate_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"데이터 생성 완료: {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"전체 샘플 수: {len(df)}")
    print(f"연체(1) 비율: {df['is_overdue'].mean() * 100:.2f}% (불균형 데이터 확인)")


if __name__ == "__main__":
    main()
