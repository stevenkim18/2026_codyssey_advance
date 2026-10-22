"""src/calculus.py 검증: 중심차분 수치 미분과 Gradient 시각화."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from src import calculus as calc


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_mission_derivative_of_x_squared_at_3():
    check = calc.check_derivative(lambda x: x**2, lambda x: 2 * x, x=3.0)
    assert check.passed
    assert check.abs_error <= 1e-4
    assert np.isclose(check.numerical, 6.0, atol=1e-4)


def test_derivative_of_other_functions():
    assert np.isclose(calc.numerical_derivative(np.sin, 0.0), 1.0, atol=1e-8)
    assert np.isclose(calc.numerical_derivative(np.exp, 1.0), np.e, atol=1e-6)
    assert np.isclose(calc.numerical_derivative(lambda x: x**3, 2.0), 12.0, atol=1e-4)


def test_central_difference_is_more_accurate_than_forward_difference():
    h = 1e-3
    central = abs(calc.numerical_derivative(np.exp, 1.0, h) - np.e)
    forward = abs((np.exp(1.0 + h) - np.exp(1.0)) / h - np.e)
    assert central < forward / 100  # O(h²) 대 O(h)


def test_gradient_matches_analytic_at_a_point_and_on_a_grid():
    point = np.array([3.0, 4.0])
    assert np.allclose(
        calc.numerical_gradient(calc.paraboloid, point), 2 * point, atol=1e-6
    )

    rng = np.random.default_rng(42)
    points = rng.uniform(-5, 5, size=(2, 50))
    assert np.allclose(
        calc.numerical_gradient(calc.paraboloid, points), 2 * points, atol=1e-6
    )


def test_gradient_of_non_symmetric_function():
    def f(p):
        return p[0] ** 2 + 10 * p[1] ** 2 + p[0] * p[1]

    grad = calc.numerical_gradient(f, np.array([1.0, 2.0]))
    assert np.allclose(grad, [2 * 1 + 2, 20 * 2 + 1], atol=1e-6)


def test_gradient_arrows_are_perpendicular_to_contours(tmp_path):
    fig, worst = calc.plot_gradient_field(save_path=tmp_path / "g.png")
    assert worst < 1e-2  # 접선과 Gradient 의 |cos| (0 이면 수직)
    assert (tmp_path / "g.png").stat().st_size > 0
    assert len(fig.axes) == 1


def test_perpendicularity_detects_a_non_perpendicular_direction():
    """검증기 자체의 민감도: 이 등고선의 함수가 아닌 f = x 의 Gradient (1, 0) 은 원의 접선과 수직이 아니다."""
    axis = np.linspace(-5, 5, 200)
    x, y = np.meshgrid(axis, axis)
    fig, ax = plt.subplots()
    contour = ax.contour(x, y, calc.paraboloid(np.stack([x, y])), levels=[4.0])

    assert calc.perpendicularity(contour, calc.paraboloid) < 1e-2
    assert calc.perpendicularity(contour, lambda p: p[0]) > 0.5
    plt.close(fig)
