"""src/linear_algebra.py 검증: 행렬식과 면적비, Power Iteration, SVD 압축."""

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from src import linear_algebra as la

MISSION_MATRIX = np.array([[4.0, 1.0], [1.0, 3.0]])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- 행렬 변환 -----------------------------------------------------------------
def test_basic_matrices():
    r = la.rotation_matrix(np.pi / 6)
    assert np.allclose(r.T @ r, np.eye(2))  # 회전은 직교 행렬
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.array_equal(la.scaling_matrix(2, 0.5), [[2.0, 0.0], [0.0, 0.5]])
    assert np.array_equal(la.shear_matrix(0.7), [[1.0, 0.7], [0.0, 1.0]])


def test_unit_circle_uses_mission_sampling():
    circle = la.unit_circle()
    assert circle.shape == (2, 100)  # np.linspace(0, 2π, 100)
    assert np.allclose(np.linalg.norm(circle, axis=0), 1.0)


def test_scaling_gives_ellipse_with_semi_axes_2_and_half():
    moved = la.apply_transform(la.scaling_matrix(2, 0.5), la.unit_circle())
    assert np.isclose(moved[0].max(), 2.0, atol=1e-3)
    assert np.isclose(moved[1].max(), 0.5, atol=1e-3)


def test_shear_tilts_circle_and_keeps_height():
    moved = la.apply_transform(la.shear_matrix(1.0), la.unit_circle())
    assert np.allclose(moved[1], la.unit_circle()[1])  # y 는 그대로
    assert moved[0].max() > 1.2  # x 가 y 에 비례해 밀린다


@pytest.mark.parametrize(
    "matrix",
    [
        la.rotation_matrix(np.pi / 4),
        la.rotation_matrix(1.234),
        la.scaling_matrix(2.0, 0.5),
        la.shear_matrix(1.0),
        la.scaling_matrix(2.0, 1.5),  # det = 3
        np.array([[1.0, 0.0], [0.0, -1.0]]),  # det = -1 (방향 뒤집힘)
        np.array([[1.0, 2.0], [2.0, 4.0]]),  # det = 0 (특이 행렬)
    ],
)
def test_area_ratio_matches_determinant_within_one_percent(matrix):
    check = la.check_area_ratio(matrix)
    assert check.passed
    assert check.rel_error <= 0.01
    assert np.isclose(check.area_ratio, np.linalg.det(matrix), atol=1e-9)


def test_area_ratio_error_is_machine_precision():
    check = la.check_area_ratio(la.scaling_matrix(2.0, 1.5))
    assert check.rel_error <= 1e-12  # 선형 변환은 다각형 면적을 정확히 det 배로 바꾼다


def test_polygon_area_of_unit_square_is_signed():
    square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    assert np.isclose(la.polygon_area(square), 1.0)
    assert np.isclose(la.polygon_area(square[:, ::-1]), -1.0)


# --- Power Iteration -----------------------------------------------------------
def test_power_iteration_on_mission_matrix():
    result = la.power_iteration(MISSION_MATRIX, tol=1e-6)
    assert result.converged
    assert np.isclose(result.eigenvalue, (7 + np.sqrt(5)) / 2, atol=1e-6)  # 4.6180
    assert np.allclose(result.eigenvector, [0.8507, 0.5257], atol=1e-4)
    assert np.isclose(np.linalg.norm(result.eigenvector), 1.0)


def test_power_iteration_is_reproducible():
    first = la.power_iteration(MISSION_MATRIX)
    second = la.power_iteration(MISSION_MATRIX)
    assert first.n_iter == second.n_iter
    assert np.array_equal(first.eigenvector, second.eigenvector)


@pytest.mark.parametrize("name", list(la.power_iteration_examples()))
def test_verify_against_numpy_eig_within_five_percent(name):
    check = la.verify_power_iteration(la.power_iteration_examples()[name])
    assert check.passed
    assert check.rel_error <= 0.05
    assert check.cosine > 0.9999


def test_power_iteration_satisfies_eigen_equation_on_random_symmetric_matrices():
    rng = np.random.default_rng(42)
    for _ in range(5):
        b = rng.normal(size=(5, 5))
        a = b @ b.T  # 양의 준정부호: 지배 고유값이 양수
        result = la.power_iteration(a, max_iter=5000, tol=1e-10)
        residual = np.linalg.norm(
            a @ result.eigenvector - result.eigenvalue * result.eigenvector
        )
        assert result.converged
        assert residual < 1e-6


def test_power_iteration_handles_negative_dominant_eigenvalue():
    result = la.power_iteration(np.diag([-5.0, 2.0]))
    assert result.converged
    assert np.isclose(result.eigenvalue, -5.0)


def test_power_iteration_reports_non_convergence_for_rotation():
    result = la.power_iteration(la.rotation_matrix(np.pi / 2), max_iter=200)
    assert not result.converged  # 복소 고유값 ±i 라서 방향이 계속 돈다


def test_power_iteration_rejects_bad_input():
    with pytest.raises(ValueError):
        la.power_iteration(np.ones((2, 3)))
    with pytest.raises(ValueError):
        la.power_iteration(np.zeros((2, 2)))


def test_power_iteration_gives_largest_singular_value_via_gram_matrix():
    image = la.make_sample_image()
    sigma1 = np.sqrt(
        la.power_iteration(image.T @ image, max_iter=5000, tol=1e-12).eigenvalue
    )
    assert np.isclose(
        sigma1, la.svd_decompose(image)[1][0], rtol=1e-6
    )  # σ₁² = λmax(AᵀA)


# --- SVD 압축 ------------------------------------------------------------------
def test_sample_image_satisfies_dataset_guide():
    image = la.make_sample_image()
    assert image.shape == (64, 64)  # 64×64 이하의 그레이스케일
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert np.array_equal(image, la.make_sample_image())  # 항상 같은 이미지


def test_full_rank_reconstruction_is_exact():
    image = la.make_sample_image()
    assert np.allclose(la.svd_compress(image, 64), image)


def test_k_larger_than_rank_is_clamped():
    image = la.make_sample_image()
    assert np.array_equal(la.svd_compress(image, 100), la.svd_compress(image, 64))
    assert la.svd_stats(image, 100).k_eff == 64


def test_rank_k_approximation_has_rank_k():
    approx = la.svd_compress(la.make_sample_image(), 10)
    assert np.linalg.matrix_rank(approx, tol=1e-8) == 10


def test_error_decreases_with_k_and_matches_discarded_energy():
    image = la.make_sample_image()
    errors = [la.svd_stats(image, k).rel_error for k in (1, 5, 10, 25, 50, 64)]
    assert all(a > b for a, b in pairwise(errors))
    for k in (10, 50, 100):
        stats = la.svd_stats(image, k)
        # Eckart–Young: ‖A − Aₖ‖² / ‖A‖² = 버린 특이값의 에너지 비율
        assert np.isclose(stats.rel_error**2, 1.0 - stats.energy, atol=1e-10)


def test_storage_ratio_formula():
    stats = la.svd_stats(np.zeros((64, 64)) + 1.0, 10)
    assert np.isclose(stats.storage, 10 * (64 + 64 + 1) / (64 * 64))


def test_svd_reconstruct_rejects_nonpositive_k():
    u, s, vt = la.svd_decompose(la.make_sample_image())
    with pytest.raises(ValueError):
        la.svd_reconstruct(u, s, vt, 0)


# --- 그래프 --------------------------------------------------------------------
def test_transformation_figure_has_three_overlaid_panels_and_saves_png(tmp_path):
    fig = la.plot_transformations(save_path=tmp_path / "t.png")
    assert len(fig.axes) == 3
    assert (tmp_path / "t.png").stat().st_size > 0


def test_svd_figure_compares_original_and_three_ranks_in_one_figure(tmp_path):
    fig = la.plot_svd_comparison(save_path=tmp_path / "svd.png")
    assert len(fig.axes) == 4  # 원본 + k = 10, 50, 100
    assert (tmp_path / "svd.png").stat().st_size > 0


def test_other_figures_render(tmp_path):
    la.plot_power_iteration(save_path=tmp_path / "p.png")
    la.plot_singular_values(save_path=tmp_path / "s.png")
    assert (tmp_path / "p.png").exists() and (tmp_path / "s.png").exists()
