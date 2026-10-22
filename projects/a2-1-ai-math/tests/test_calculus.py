import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import calculus as calc
from src import optimizer as opt


def test_central_difference_of_square_at_three_is_within_1e_4_of_six():
    result = calc.verify_derivative(lambda x: x * x, exact=6.0, x=3.0)
    assert result["abs_error"] < 1e-4
    assert result["passed"]
    assert result["numeric"] == pytest.approx(6.0, abs=1e-4)


def test_central_difference_is_more_accurate_than_forward_difference():
    f = np.exp
    exact = np.e
    h = 1e-3
    central = abs(calc.central_difference(f, 1.0, h) - exact)
    forward = abs(calc.forward_difference(f, 1.0, h) - exact)
    assert central < forward / 100


def test_central_difference_error_scales_as_h_squared():
    exact = np.cos(1.0)
    e1 = abs(calc.central_difference(np.sin, 1.0, 1e-2) - exact)
    e2 = abs(calc.central_difference(np.sin, 1.0, 1e-3) - exact)
    assert e1 / e2 == pytest.approx(
        100.0, rel=0.05
    )  # h가 10배 작아지면 오차는 100배 작아진다


def test_numerical_gradient_matches_analytic_gradient():
    point = np.array([1.0, 2.0])
    assert calc.numerical_gradient(calc.paraboloid, point) == pytest.approx(
        calc.paraboloid_gradient(point), abs=1e-6
    )


def test_numerical_hessian_of_paraboloid_and_rosenbrock():
    assert calc.numerical_hessian(calc.paraboloid, [0.5, -1.0]) == pytest.approx(
        np.diag([2.0, 2.0]), abs=1e-5
    )
    point = np.array([-1.2, 1.0])
    assert calc.numerical_hessian(opt.ROSENBROCK.f, point, h=1e-4) == pytest.approx(
        opt.ROSENBROCK.hess(point), rel=1e-4
    )


def test_gradient_is_perpendicular_to_contour_tangent():
    assert calc.max_gradient_contour_cosine() < 1e-6


def test_gradient_points_in_direction_of_steepest_increase():
    point = np.array([1.0, 2.0])
    grad = calc.numerical_gradient(calc.paraboloid, point)
    direction = grad / np.linalg.norm(grad)
    step = 1e-3
    gains = []
    for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        d = np.array([np.cos(angle), np.sin(angle)])
        gains.append(calc.paraboloid(point + step * d))
    assert calc.paraboloid(point + step * direction) >= max(gains) - 1e-12


def test_contour_points_lie_on_requested_circles():
    points = calc.contour_points((1.0, 2.0), n_angles=6)
    assert points.shape == (12, 2)
    assert np.linalg.norm(points[:6], axis=1) == pytest.approx(np.ones(6))
    assert np.linalg.norm(points[6:], axis=1) == pytest.approx(2 * np.ones(6))


def test_plot_functions_return_figures_without_saving():
    figures = [
        calc.plot_gradient_field(filename=None),
        calc.plot_difference_error(filename=None),
    ]
    assert all(isinstance(fig, plt.Figure) for fig in figures)
    plt.close("all")
