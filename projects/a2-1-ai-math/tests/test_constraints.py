"""미션 제약 사항(허용 라이브러리, eig 사용 범위, Docstring, 시드, 산출물 구성) 정적 검증."""

import ast
import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FILES = sorted((PROJECT_ROOT / "src").glob("*.py"))
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
FORBIDDEN = {"torch", "tensorflow", "jax", "sklearn", "scipy", "autograd", "pandas"}


def notebook_code(path: Path) -> str:
    """노트북의 코드 셀을 하나의 문자열로 합친다."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def parsed_sources() -> list[tuple[str, ast.Module]]:
    """(이름, AST) 목록: src/*.py 와 노트북 코드 셀."""
    items = [(p.name, ast.parse(p.read_text(encoding="utf-8"))) for p in SRC_FILES]
    items += [(p.name, ast.parse(notebook_code(p))) for p in NOTEBOOKS]
    return items


def test_required_files_exist():
    required = [
        "src/linear_algebra.py",
        "src/calculus.py",
        "src/optimizer.py",
        "notebooks/backprop_derivation.ipynb",
        "notebooks/probability_loss.ipynb",
        "README.md",
        "requirements.txt",
    ]
    for rel in required:
        assert (PROJECT_ROOT / rel).is_file(), rel


@pytest.mark.parametrize(("name", "tree"), parsed_sources())
def test_no_forbidden_libraries(name, tree):
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & FORBIDDEN, f"{name}: {imported & FORBIDDEN}"


def eig_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """`linalg.eig` 호출을 (감싸는 함수 이름, 줄 번호) 로 모은다."""
    found = []

    def visit(node, enclosing):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, ast.FunctionDef) else enclosing
            if isinstance(child, ast.Attribute) and child.attr in {
                "eig",
                "eigh",
                "eigvals",
                "eigvalsh",
            }:
                found.append((enclosing, child.lineno))
            visit(child, name)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize(("name", "tree"), parsed_sources())
def test_eig_is_used_only_inside_verification_functions(name, tree):
    for enclosing, lineno in eig_calls(tree):
        assert enclosing.startswith("verify_"), (
            f"{name}:{lineno} 의 eig 가 {enclosing} 안에서 쓰였다"
        )


def test_power_iteration_does_not_call_eig():
    tree = ast.parse(
        (PROJECT_ROOT / "src" / "linear_algebra.py").read_text(encoding="utf-8")
    )
    callers = {enclosing for enclosing, _ in eig_calls(tree)}
    assert callers == {"verify_power_iteration"}


@pytest.mark.parametrize(("name", "tree"), parsed_sources())
def test_every_function_and_class_has_a_docstring(name, tree):
    missing = [
        f"{node.name} (line {node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
        and ast.get_docstring(node) is None
    ]
    assert not missing, f"{name}: {missing}"


@pytest.mark.parametrize("path", SRC_FILES + NOTEBOOKS, ids=lambda p: p.name)
def test_seed_is_fixed_to_42(path):
    if path.name == "__init__.py":
        pytest.skip("패키지 표식 파일")
    text = (
        notebook_code(path)
        if path.suffix == ".ipynb"
        else path.read_text(encoding="utf-8")
    )
    assert re.search(r"np\.random\.seed\(\s*(42|SEED)\s*\)", text)
    if "np.random.seed(SEED)" in text:
        assert "SEED = 42" in text


def test_requirements_pin_the_libraries_in_use():
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pinned = {line.split("==")[0].lower() for line in lines if "==" in line}
    assert {"numpy", "matplotlib", "seaborn"} <= pinned


def test_expected_pngs_are_saved_in_outputs():
    expected = [
        "linear_transformations.png",
        "power_iteration_convergence.png",
        "svd_compression.png",
        "svd_singular_values.png",
        "gradient_contour.png",
        "gd_convergence.png",
        "lr_divergence.png",
        "momentum_vs_gd.png",
        "backprop_training.png",
        "normal_pdf.png",
        "bernoulli_pmf.png",
        "mse_mle.png",
        "cross_entropy_mle.png",
    ]
    for name in expected:
        assert (PROJECT_ROOT / "outputs" / name).stat().st_size > 0, name
