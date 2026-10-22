import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import linear_algebra as la


# ---------------------------------------------------------------- 기하 변환
def test_rotation_is_orthogonal_with_unit_determinant():
    rot = la.rotation_matrix(np.deg2rad(30))
    assert rot.T @ rot == pytest.approx(np.eye(2))
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_scaling_and_shear_determinants():
    assert np.linalg.det(la.scaling_matrix(2.0, 0.5)) == pytest.approx(1.0)
    assert np.linalg.det(la.scaling_matrix(3.0, 2.0)) == pytest.approx(6.0)
    assert np.linalg.det(la.shear_matrix(0.7)) == pytest.approx(1.0)


def test_unit_circle_points_have_radius_one():
    circle = la.unit_circle(100)
    assert circle.shape == (2, 100)
    assert np.linalg.norm(circle, axis=0) == pytest.approx(np.ones(100))


def test_polygon_area_of_circle_is_close_to_pi():
    assert la.polygon_area(la.unit_circle(360)) == pytest.approx(np.pi, rel=1e-3)


def test_area_ratio_matches_determinant_within_one_percent_for_all_default_transforms():
    for name, matrix in la.default_transforms().items():
        check = la.check_area_ratio(matrix)
        assert check["rel_error"] < 0.01, name
        assert check["ratio"] == pytest.approx(abs(check["det"]), abs=1e-9)


def test_area_ratio_scales_with_determinant_and_collapses_when_singular():
    assert la.check_area_ratio(la.scaling_matrix(1.5, 2.0))["ratio"] == pytest.approx(
        3.0
    )
    assert la.check_area_ratio(np.array([[1.0, 2.0], [2.0, 4.0]]))[
        "ratio"
    ] == pytest.approx(0.0, abs=1e-12)


def test_matrix_product_applies_right_factor_first_and_order_matters():
    rot = la.rotation_matrix(np.pi / 4)
    scale = la.scaling_matrix(2.0, 0.5)
    point = np.array([[1.0], [0.0]])
    stepwise = la.apply_transform(rot, la.apply_transform(scale, point))
    assert la.apply_transform(rot @ scale, point) == pytest.approx(stepwise)
    assert not np.allclose(rot @ scale, scale @ rot)


# ---------------------------------------------------------------- Power Iteration
def test_power_iteration_matches_numpy_eig_within_five_percent():
    for name, matrix in la.power_iteration_examples().items():
        result = la.verify_power_iteration(matrix)
        assert result["rel_error"] < 0.05, name
        assert result["passed"]
        assert result["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_power_iteration_known_eigenvalue():
    value, vector, history = la.power_iteration(np.array([[4.0, 1.0], [2.0, 3.0]]))
    assert value == pytest.approx(5.0)
    assert vector == pytest.approx(
        np.array([1.0, 1.0]) / np.sqrt(2), abs=1e-6
    ) or vector == pytest.approx(-np.array([1.0, 1.0]) / np.sqrt(2), abs=1e-6)
    assert history[-1] == pytest.approx(value)


def test_power_iteration_handles_negative_dominant_eigenvalue():
    value, _, _ = la.power_iteration(np.diag([-5.0, 2.0]))
    assert value == pytest.approx(-5.0)


def test_power_iteration_is_reproducible_with_fixed_seed():
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    first = la.power_iteration(matrix, num_iters=5, tol=0.0, seed=42)[2]
    second = la.power_iteration(matrix, num_iters=5, tol=0.0, seed=42)[2]
    assert np.array_equal(first, second)


def test_power_iteration_zero_matrix_returns_zero():
    value, _, _ = la.power_iteration(np.zeros((3, 3)))
    assert value == 0.0


# ---------------------------------------------------------------- SVD
@pytest.fixture(scope="module")
def image():
    return la.load_sample_image()


def test_sample_image_is_grayscale_in_unit_range(image):
    assert image.ndim == 2
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_full_rank_reconstruction_is_exact(image):
    u, s, vt = la.svd_decompose(image)
    assert la.svd_reconstruct(u, s, vt, len(s)) == pytest.approx(image, abs=1e-10)


def test_rank_k_approximation_has_rank_k(image):
    u, s, vt = la.svd_decompose(image)
    assert np.linalg.matrix_rank(la.svd_reconstruct(u, s, vt, 10)) == 10


def test_error_decreases_as_k_grows_and_matches_singular_value_formula(image):
    u, s, vt = la.svd_decompose(image)
    errors = []
    for k in (10, 50, 100):
        approx = la.svd_reconstruct(u, s, vt, k)
        error = la.relative_error(image, approx)
        # Eckart-Young: 잘라낸 특이값의 에너지가 곧 오차다
        assert error == pytest.approx(
            np.sqrt(np.sum(s[k:] ** 2) / np.sum(s**2)), rel=1e-9
        )
        errors.append(error)
    assert errors[0] > errors[1] > errors[2]
    assert la.psnr(image, la.svd_reconstruct(u, s, vt, 100)) > la.psnr(
        image, la.svd_reconstruct(u, s, vt, 10)
    )


def test_storage_ratio_and_energy_ratio(image):
    assert la.storage_ratio((600, 512), 100) == pytest.approx(
        100 * (600 + 512 + 1) / (600 * 512)
    )
    _, s, _ = la.svd_decompose(image)
    assert la.energy_ratio(s, len(s)) == pytest.approx(1.0)
    assert la.energy_ratio(s, 10) < la.energy_ratio(s, 50) < la.energy_ratio(s, 100)


def test_svd_compress_output_is_displayable(image):
    approx = la.svd_compress(image, 10)
    assert approx.shape == image.shape
    assert approx.min() >= 0.0 and approx.max() <= 1.0


# ---------------------------------------------------------------- 시각화
def test_plot_functions_return_figures_without_saving(image):
    figures = [
        la.plot_transformations(filename=None),
        la.plot_power_iteration(filename=None),
        la.plot_svd_comparison(image, filename=None),
        la.plot_singular_values(image, filename=None),
    ]
    assert all(isinstance(fig, plt.Figure) for fig in figures)
    plt.close("all")
