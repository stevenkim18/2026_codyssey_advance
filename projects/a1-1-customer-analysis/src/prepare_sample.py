"""H&M 원본 CSV에서 고객 N명을 무작위로 뽑아 분석용 샘플을 만든다.

사용 예 (프로젝트 루트에서):
    uv run python src/prepare_sample.py --n-customers 5000 --seed 42

입력 : data/raw/{articles,customers,transactions_train}.csv
출력 : data/sample/{customers,transactions,articles}_sample.csv
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
CHUNK_ROWS = 2_000_000


def sample_customers(n_customers: int, seed: int) -> pd.DataFrame:
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    return customers.sample(n=n_customers, random_state=seed).reset_index(drop=True)


def filter_transactions(customer_ids: set[str]) -> pd.DataFrame:
    """대용량 거래 파일을 청크로 읽으며 샘플 고객의 행만 모은다."""
    parts = []
    reader = pd.read_csv(
        RAW_DIR / "transactions_train.csv",
        dtype={"article_id": str},
        chunksize=CHUNK_ROWS,
    )
    for i, chunk in enumerate(reader, start=1):
        parts.append(chunk[chunk["customer_id"].isin(customer_ids)])
        print(f"  chunk {i}: 누적 {sum(len(p) for p in parts):,}행")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    customers = sample_customers(args.n_customers, args.seed)
    transactions = filter_transactions(set(customers["customer_id"]))

    # 거래가 없는 고객은 RFM 대상이 아니므로 제외
    customers = customers[customers["customer_id"].isin(transactions["customer_id"])]

    articles = pd.read_csv(RAW_DIR / "articles.csv", dtype={"article_id": str})
    articles = articles[articles["article_id"].isin(transactions["article_id"])]

    customers.to_csv(SAMPLE_DIR / "customers_sample.csv", index=False)
    transactions.to_csv(SAMPLE_DIR / "transactions_sample.csv", index=False)
    articles.to_csv(SAMPLE_DIR / "articles_sample.csv", index=False)

    print(
        f"고객 {len(customers):,}명 / 거래 {len(transactions):,}행 / 상품 {len(articles):,}개"
    )
    print(f"거래 기간: {transactions['t_dat'].min()} ~ {transactions['t_dat'].max()}")


if __name__ == "__main__":
    main()
