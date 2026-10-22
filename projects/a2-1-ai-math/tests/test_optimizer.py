import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import calculus as calc
from src import optimizer as opt


def run(optimizer, objective=opt.SPHERE, n_iters=100):
    return opt.minimize(optimizer, objective, opt.START, n_iters)


# ---------------------------------------------------------------- Vanilla GD
def test_vanilla_gd_lr_point_one_reaches_radius_under_point_one_after_100_steps():
    trajectory = run(opt.VanillaGD(0.1))
    assert trajectory.radius[-1] < 0.1
    assert trajectory.radius[-1] == pytest.approx(
        5.0 * 0.8**100
    )  # x ← 0.8x 라서 정확히 예측된다
    assert not trajectory.diverged


def test_vanilla_gd_update_rule():
    x = np.array([4.0, 3.0])
    assert opt.VanillaGD(0.1).step(x, opt.SPHERE.grad(x)) == pytest.approx(0.8 * x)


def test_learning_rate_regimes_on_sphere():
    """f=x²+y²에서 x ← (1-2lr)x 이므로 lr=0.5는 1회에 도달, 1.0은 진동, 1.1은 발산."""
    assert run(opt.VanillaGD(0.5), n_iters=1).radius[-1] == pytest.approx(
        0.0, abs=1e-12
    )
    oscillating = run(opt.VanillaGD(1.0))
    assert oscillating.radius[-1] == pytest.approx(5.0)
    assert not oscillating.diverged
    assert run(opt.VanillaGD(1.1), n_iters=200).diverged


@pytest.mark.parametrize("lr", [0.5, 1.0])
def test_learning_rate_at_least_half_diverges_on_ill_conditioned(lr):
    """f=x²+10y²는 y축 계수가 20이라 lr > 0.1이면 발산한다. 미션의 'lr≥0.5 발산'이 성립하는 함수다."""
    assert run(opt.VanillaGD(lr), opt.ILL_CONDITIONED).diverged
    assert run(opt.Momentum(lr, 0.9), opt.ILL_CONDITIONED).diverged


def test_divergence_stops_the_run_early_without_overflow_warnings():
    with np.errstate(all="raise"):
        trajectory = opt.minimize(
            opt.VanillaGD(5.0), opt.ILL_CONDITIONED, opt.START, 1000
        )
    assert trajectory.diverged
    assert len(trajectory.path) < 100


# ---------------------------------------------------------------- Momentum
def test_momentum_first_step_equals_gd_and_second_step_uses_velocity():
    x = np.array([4.0, 3.0])
    momentum = opt.Momentum(lr=0.1, beta=0.9)
    g1 = opt.SPHERE.grad(x)
    x1 = momentum.step(x, g1)
    assert x1 == pytest.approx(x - 0.1 * g1)
    g2 = opt.SPHERE.grad(x1)
    x2 = momentum.step(x1, g2)
    assert x2 == pytest.approx(x1 - 0.1 * (0.9 * g1 + g2))


def test_momentum_reset_clears_velocity():
    momentum = opt.Momentum(0.1)
    momentum.step(np.ones(2), np.ones(2))
    momentum.reset()
    assert momentum.velocity is None


def test_momentum_beats_gd_on_ill_conditioned_function_at_small_learning_rate():
    gd = opt.iterations_to_converge(lambda: opt.VanillaGD(0.02), opt.ILL_CONDITIONED)
    mom = opt.iterations_to_converge(
        lambda: opt.Momentum(0.02, 0.9), opt.ILL_CONDITIONED
    )
    assert mom < gd


def test_momentum_stays_stable_beyond_gd_stability_limit():
    """GD는 lr > 0.1에서 발산하지만 Momentum(β=0.9)은 lr < 2(1+β)/20 = 0.19까지 안정하다."""
    assert run(opt.VanillaGD(0.15), opt.ILL_CONDITIONED, 200).diverged
    assert not run(opt.Momentum(0.15, 0.9), opt.ILL_CONDITIONED, 400).diverged
    assert run(opt.Momentum(0.15, 0.9), opt.ILL_CONDITIONED, 400).radius[-1] < 1e-3


def test_tuned_vanilla_gd_beats_momentum_on_quadratic():
    """학습률을 튜닝하면 이 2차 함수에서는 GD가 β=0.9 Momentum보다 빠르다 (한계를 README에 적은 근거)."""
    _, gd = opt.best_lr(*opt.sweep_iterations(lambda lr: opt.VanillaGD(lr)))
    _, mom = opt.best_lr(*opt.sweep_iterations(lambda lr: opt.Momentum(lr, 0.9)))
    assert gd < mom


# ---------------------------------------------------------------- Adam, Newton (보너스)
def test_adam_first_step_has_magnitude_lr_per_coordinate():
    adam = opt.Adam(lr=0.1)
    x = np.array([4.0, 3.0])
    new_x = adam.step(x, opt.ILL_CONDITIONED.grad(x))
    assert x - new_x == pytest.approx(
        np.array([0.1, 0.1]), abs=1e-6
    )  # 편향 보정 후 m̂/√ŝ = ±1


def test_adam_converges_on_ill_conditioned_function():
    assert np.isfinite(
        opt.iterations_to_converge(lambda: opt.Adam(0.1), opt.ILL_CONDITIONED)
    )


def test_adam_reset_restarts_time_step():
    adam = opt.Adam()
    adam.step(np.ones(2), np.ones(2))
    adam.reset()
    assert adam.t == 0 and adam.m is None


def test_newton_solves_quadratic_in_one_step():
    for objective in (opt.SPHERE, opt.ILL_CONDITIONED):
        trajectory = opt.minimize(opt.Newton(objective.hess), objective, opt.START, 3)
        assert trajectory.radius[1] == pytest.approx(0.0, abs=1e-12)


def test_newton_converges_on_rosenbrock():
    trajectory = opt.minimize(
        opt.Newton(opt.ROSENBROCK.hess), opt.ROSENBROCK, (-1.2, 1.0), 50
    )
    assert trajectory.path[-1] == pytest.approx([1.0, 1.0], abs=1e-8)


def test_analytic_hessians_match_numerical_hessian():
    point = np.array([0.7, -0.4])
    for objective in (opt.SPHERE, opt.ILL_CONDITIONED, opt.ROSENBROCK):
        assert calc.numerical_hessian(objective.f, point) == pytest.approx(
            objective.hess(point), rel=1e-4, abs=1e-5
        )


def test_analytic_gradients_match_numerical_gradient():
    point = np.array([0.7, -0.4])
    for objective in (opt.SPHERE, opt.ILL_CONDITIONED, opt.ROSENBROCK):
        assert calc.numerical_gradient(objective.f, point) == pytest.approx(
            objective.grad(point), rel=1e-5, abs=1e-6
        )


# ---------------------------------------------------------------- 보조
def test_minimize_records_start_and_values():
    trajectory = run(opt.VanillaGD(0.1), n_iters=5)
    assert trajectory.path.shape == (6, 2)
    assert trajectory.path[0].tolist() == list(opt.START)
    assert trajectory.values[0] == pytest.approx(25.0)


def test_iterations_to_converge_returns_nan_when_it_never_converges():
    assert np.isnan(
        opt.iterations_to_converge(lambda: opt.VanillaGD(1.0), opt.SPHERE, max_iters=50)
    )


def test_plot_functions_return_figures_without_saving():
    figures = [
        opt.plot_gd_convergence(filename=None),
        opt.plot_lr_sweep(filename=None),
        opt.plot_momentum_vs_gd(filename=None),
        opt.plot_optimizer_comparison(filename=None),
    ]
    assert all(isinstance(fig, plt.Figure) for fig in figures)
    plt.close("all")
