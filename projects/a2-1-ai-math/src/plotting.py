"""그래프 공통 설정: 한글 글꼴, 색상 팔레트, 출력 경로 (A2-1).

모든 모듈이 같은 모양의 그래프를 내도록 `setup_style()`과 `save_figure()`를 공유한다.
"""

from __future__ import annotations

import platform
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEED = 42

# 범주형 색: 파랑, 주황, 청록, 노랑 순서로 고정한다 (시리즈에 색을 한 번 배정하면 바꾸지 않는다).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def setup_style() -> None:
    """한글 글꼴, 눈금/격자 색, 배경색을 한 번에 설정한다."""
    font = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(
        platform.system(), "NanumGothic"
    )
    plt.rcParams.update(
        {
            # 로그 축 눈금(10⁻⁵)의 `−`는 한글 글꼴에 없으므로 DejaVu를 먼저 두고 한글은 대체 글꼴로 그린다.
            "font.family": ["DejaVu Sans", font],
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, name: str, output_dir: Path | None = None) -> Path:
    """그림을 `outputs/<name>`에 PNG로 저장하고 경로를 돌려준다.

    Args:
        fig: 저장할 Figure
        name: 파일 이름 (예: ``"transformations.png"``)
        output_dir: 저장 폴더. 없으면 만든다. 기본값은 프로젝트의 ``outputs/``.
    """
    folder = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path)
    return path
