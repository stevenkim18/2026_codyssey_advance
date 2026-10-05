"""data/images 의 상품 이미지를 읽어 고정 크기 NumPy 배열로 저장한다.

처리 순서: imread -> 중앙 자르기(고정 크기) -> arr[::stride, ::stride] 다운샘플링
(미션 허용 범위: imread, NumPy 슬라이싱. Pillow/OpenCV는 사용하지 않는다.)

사용 예 (프로젝트 루트에서):
    uv run python src/build_image_array.py --stride 10

출력 : data/processed/image_arrays.npy      (N, H, W, 3) uint8
       data/processed/image_article_ids.csv  배열 N개와 같은 순서의 article_id
"""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
CROP_H, CROP_W = 1750, 1160  # 원본(1750x1166)의 중앙을 자른 크기


def center_crop(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """중앙 기준으로 (height, width)로 자른다. 원본이 더 작으면 0으로 채운다."""
    h, w = arr.shape[:2]
    out = np.zeros((height, width, arr.shape[2]), dtype=arr.dtype)
    src_top, src_left = max((h - height) // 2, 0), max((w - width) // 2, 0)
    dst_top, dst_left = max((height - h) // 2, 0), max((width - w) // 2, 0)
    copy_h, copy_w = min(h, height), min(w, width)
    out[dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = arr[
        src_top : src_top + copy_h, src_left : src_left + copy_w
    ]
    return out


def load_image(path: Path, stride: int) -> np.ndarray:
    arr = mpimg.imread(path)
    if arr.dtype != np.uint8:  # PNG 등은 0~1 float로 읽힌다
        arr = (arr * 255).round().astype(np.uint8)
    if arr.ndim == 2:  # 흑백이면 3채널로 확장
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    arr = arr[:, :, :3]  # 알파 채널 제거
    return center_crop(arr, CROP_H, CROP_W)[::stride, ::stride]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=10, help="다운샘플링 간격")
    args = parser.parse_args()

    paths = sorted(IMAGE_DIR.rglob("*.jpg"))
    if not paths:
        raise SystemExit(
            "data/images 에 이미지가 없습니다. download_images.py 를 먼저 실행하세요."
        )

    arrays = np.stack([load_image(p, args.stride) for p in paths])
    article_ids = [p.stem for p in paths]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "image_arrays.npy", arrays)
    pd.DataFrame({"article_id": article_ids}).to_csv(
        OUT_DIR / "image_article_ids.csv", index=False
    )
    size_mb = arrays.nbytes / 1e6
    print(
        f"이미지 {len(paths):,}장 -> 배열 {arrays.shape}, {arrays.dtype}, {size_mb:.1f}MB"
    )


if __name__ == "__main__":
    main()
