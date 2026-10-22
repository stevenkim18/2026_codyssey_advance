"""미션 제약 사항(7장)을 소스 코드 수준에서 확인한다."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src import linear_algebra as la

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN_IMPORTS = {"torch", "tensorflow", "jax", "sklearn", "autograd", "scipy"}


def source_files():
    return sorted(path for path in SRC.glob("*.py"))


def test_no_autodiff_or_sklearn_imports():
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not roots & FORBIDDEN_IMPORTS, (
                f"{path.name}: {roots & FORBIDDEN_IMPORTS}"
            )


def test_linalg_eig_is_called_only_in_the_power_iteration_reference_helper():
    callers = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr in {
                    "eig",
                    "eigh",
                    "eigvals",
                    "eigvalsh",
                }:
                    callers.append((path.name, func.name))
    assert set(callers) == {("linear_algebra.py", "reference_dominant_eigenpair")}


def test_every_module_function_and_class_has_a_docstring():
    missing = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(f"{path.name} (module)")
        for node in ast.walk(tree):
            is_definition = isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            )
            if is_definition and ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.name}")
    assert not missing, missing


def test_reference_eigenpair_matches_known_matrix():
    value, vector = la.reference_dominant_eigenpair(np.array([[4.0, 1.0], [2.0, 3.0]]))
    assert value == pytest.approx(5.0)
    assert abs(vector @ np.array([1.0, 1.0]) / np.sqrt(2)) == pytest.approx(1.0)
