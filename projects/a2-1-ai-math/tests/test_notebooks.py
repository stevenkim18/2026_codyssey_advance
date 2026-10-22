"""노트북이 처음부터 끝까지 오류 없이 실행되고(셀 안의 assert 통과), 저장된 결과가 있는지 확인한다."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NAMES = ["backprop_derivation.ipynb", "probability_loss.ipynb"]


@pytest.mark.parametrize("name", NAMES)
def test_notebook_has_saved_outputs(name):
    nb = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)


@pytest.mark.parametrize("name", NAMES)
def test_notebook_executes_without_error(name, monkeypatch):
    monkeypatch.delenv(
        "MPLBACKEND", raising=False
    )  # 노트북 커널은 inline 백엔드를 쓴다
    nb = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()  # 셀 안의 assert 가 실패하면 CellExecutionError


def test_backprop_notebook_covers_required_derivation_items():
    text = " ".join(
        c.source
        for c in nbformat.read(NOTEBOOK_DIR / NAMES[0], as_version=4).cells
        if c.cell_type == "markdown"
    )
    for item in (
        "W_1",
        "W_2",
        "a_1",
        "z_1",
        "z_2",
        "체크포인트",
        "shape",
        "(2, 2)",
        "(2,)",
        "연쇄 법칙",
    ):
        assert item in text, item


def test_probability_notebook_covers_both_mle_derivations():
    text = " ".join(
        c.source
        for c in nbformat.read(NOTEBOOK_DIR / NAMES[1], as_version=4).cells
        if c.cell_type == "markdown"
    )
    for item in (
        "MSE",
        "정규분포",
        "MLE",
        "Cross-Entropy",
        "베르누이",
        "카테고리",
        "N(0, 1)",
        "B(0.3)",
        "B(0.7)",
    ):
        assert item in text, item
