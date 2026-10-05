"""샘플 거래에서 인기 상위 N개 상품의 이미지를 Kaggle에서 내려받는다.

사전 준비: Kaggle 대회 약관 동의 + 환경변수 KAGGLE_API_TOKEN (또는 ~/.kaggle 인증)
사용 예 (프로젝트 루트에서):
    uv run python src/download_images.py --top-n 3000
이미 받은 파일은 건너뛰므로 중단 후 다시 실행해도 된다.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "h-and-m-personalized-fashion-recommendations"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
IMAGE_INDEX = PROJECT_ROOT / "data" / "raw" / "image_index.txt"
MAX_RETRIES = 6


def existing_image_ids() -> set[str] | None:
    """Kaggle 파일 목록(image_index.txt)이 있으면 실제 존재하는 이미지 id 집합을 반환."""
    if not IMAGE_INDEX.exists():
        return None
    names = IMAGE_INDEX.read_text().splitlines()
    return {n.split("/")[-1][:-4] for n in names if n.startswith("images/")}


def top_article_ids(top_n: int) -> list[str]:
    """이미지가 존재하는 상품 중 거래가 많은 순으로 top_n개."""
    t = pd.read_csv(SAMPLE_DIR / "transactions_sample.csv", dtype={"article_id": str})
    ranked = t["article_id"].value_counts().index
    available = existing_image_ids()
    if available is not None:
        ranked = [a for a in ranked if a in available]
    return list(ranked[:top_n])


def remote_path(article_id: str) -> str:
    return f"images/{article_id[:3]}/{article_id}.jpg"


def download_one(api: KaggleApi, article_id: str) -> tuple[str, str]:
    target_dir = IMAGE_DIR / article_id[:3]
    target = target_dir / f"{article_id}.jpg"
    if target.exists():
        return article_id, "skip"
    target_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RETRIES):
        try:
            api.competition_download_file(
                COMPETITION, remote_path(article_id), path=str(target_dir), quiet=True
            )
            return article_id, "ok" if target.exists() else "fail: no file"
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return article_id, "missing"
            if status == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(5 * 2**attempt)  # 5, 10, 20, 40, 80초
                continue
            return article_id, f"fail: {type(exc).__name__} {status}"
    return article_id, "fail: retries exhausted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    ids = top_article_ids(args.top_n)
    counts: dict[str, int] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, api, a) for a in ids]
        for i, fut in enumerate(as_completed(futures), start=1):
            article_id, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key in ("fail", "missing"):
                failed.append(f"{article_id} ({status})")
            if i % 200 == 0:
                print(f"  {i}/{len(ids)} {counts}")

    print("완료:", counts)
    if failed:
        (IMAGE_DIR / "failed.txt").write_text("\n".join(failed))
        print(f"실패 {len(failed)}건 → data/images/failed.txt")


if __name__ == "__main__":
    main()
