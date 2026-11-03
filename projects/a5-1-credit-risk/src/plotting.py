"""그래프 공통 스타일 (A5-1). 회귀·분류 그래프가 같은 색과 선 굵기를 쓰도록 모았다.

규칙
    - 색은 의미를 따라 고정한다. 비교 대상(모델, 변수)에는 범주형 색 3개를 정해진 순서로 쓰고,
      강조할 대상이 아닌 것은 회색으로 물린다. 크기를 나타내는 값은 파랑 한 가지 색의 명도로 그린다.
    - 선은 2px 안팎으로 얇게, 격자선은 연한 실선으로 둔다. 상자·그림자 같은 장식은 쓰지 않는다.
    - 그래프 안의 글자는 영어로 쓴다 (기본 글꼴에 한글이 없어서 깨진다). 한국어 설명은 README 에 있다.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.preprocess import OUTPUT_DIR

matplotlib.use("Agg")  # 화면 없이 파일로만 저장한다

# 바탕과 글자
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 범주형 색 (앞의 3개는 모든 쌍 조합에서 색각 이상 검증을 통과한다)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CATEGORICAL = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "s", "^"]  # 색만으로 구분하지 않도록 마커 모양도 다르게 한다

# 파랑 순차 램프 (밝음 → 어두움): Confusion Matrix 처럼 크기를 색으로 나타낼 때 쓴다
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    """모든 그래프에 같은 스타일을 적용한다 (각 모듈의 main 맨 앞에서 한 번 호출)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 150,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig: Figure, name: str) -> Path:
    """`outputs/<name>` 에 저장하고 figure 를 닫는다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
