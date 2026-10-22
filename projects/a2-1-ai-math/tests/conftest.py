import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 그림을 만든다

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
