import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # matplotlib 를 import 하기 전에 화면 없는 백엔드로 고정한다
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.data_gen import generate_data
from src.preprocess import split_data


@pytest.fixture(scope="session")
def df():
    """CSV 파일 없이도 테스트가 돌도록 생성 함수로 직접 만든다 (CSV 는 .gitignore 대상)."""
    return generate_data()


@pytest.fixture(scope="session")
def split(df):
    return split_data(df)
