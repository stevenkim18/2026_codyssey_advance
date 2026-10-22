import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLBACKEND", "Agg"
)  # matplotlib 를 import 하기 전에 화면 없는 백엔드로 고정한다
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
