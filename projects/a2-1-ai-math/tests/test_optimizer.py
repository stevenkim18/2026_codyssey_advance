"""src/optimizer.py 검증: VanillaGD, Momentum, 수렴·발산, 시각화."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from src import optimizer as opt
from src.calculus import numerical_gradient


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def run(optimizer, objective, n_iter=100):
    """시작점 (5, 5) 에서 n_iter 번 갱신한 경로를 돌려준다."""
    return opt.minimize(optimizer, objective, n_iter=n_iter)


# --- 옵티마이저 한 걸음 --------------------------------------------------------
def test_vanilla_gd_step_formula_and_input_untouched():
    params, grads = np.array([5.0, 5.0]), np.array([10.0, 10.0])
    new = opt.VanillaGD(0.1).step(params, grads)
    assert np.allclose(new, [4.0, 4.0])  # θ − η∇f
    assert np.array_equal(params, [5.0, 5.0])


def test_momentum_follows_velocity_recurrence():
    m = opt.Momentum(lr=0.1, beta=0.9)
    g1, g2 = np.array([2.0, 4.0]), np.array([1.0, -1.0])
    p1 = m.step(np.array([1.0, 1.0]), g1)  # v1 = g1
    assert np.allclose(m.velocity, g1)
    assert np.allclose(p1, [1.0, 1.0] - 0.1 * g1)
    p2 = m.step(p1, g2)  # v2 = 0.9 v1 + g2
    assert np.allclose(m.velocity, 0.9 * g1 + g2)
    assert np.allclose(p2, p1 - 0.1 * (0.9 * g1 + g2))


def test_momentum_with_zero_beta_equals_vanilla_gd():
    gd = run(opt.VanillaGD(0.05), opt.ELLIPSE, 50).path
    mo = run(opt.Momentum(0.05, beta=0.0), opt.ELLIPSE, 50).path
    assert np.allclose(gd, mo)


def test_reset_clears_velocity_so_runs_are_repeatable():
    m = opt.Momentum(0.05)
    first = run(m, opt.ELLIPSE, 30).path
    second = run(m, opt.ELLIPSE, 30).path  # minimize 가 reset 을 호출한다
    assert np.array_equal(first, second)


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        opt.VanillaGD(0.0)
    with pytest.raises(ValueError):
        opt.Momentum(-0.1)
    with pytest.raises(ValueError):
        opt.Momentum(0.1, beta=1.0)


def test_analytic_gradients_match_numerical_gradients():
    rng = np.random.default_rng(42)
    for objective in (opt.SPHERE, opt.ELLIPSE):
        for p in rng.uniform(-5, 5, size=(5, 2)):
            assert np.allclose(
                objective.grad(p), numerical_gradient(objective.f, p), atol=1e-5
            )


# --- 미션 요구 수치 ------------------------------------------------------------
def test_vanilla_gd_reaches_origin_within_radius_0_1_after_100_iterations():
    result = run(opt.VanillaGD(0.1), opt.SPHERE, 100)
    assert not result.diverged
    assert opt.radii(result.path)[-1] < 0.1
    # x ← (1 − 2·lr) x = 0.8 x 이므로 100번 뒤 반지름은 0.8¹⁰⁰ · ‖start‖
    assert np.isclose(opt.radii(result.path)[-1], 0.8**100 * np.hypot(5, 5))
    assert opt.iterations_to_reach(result.path, 0.1) == 20


@pytest.mark.parametrize("lr", [0.5, 1.0, 1.1, 2.0])
def test_ellipse_diverges_for_lr_at_least_0_5(lr):
    assert run(opt.VanillaGD(lr), opt.ELLIPSE).diverged  # 안정 조건 lr < 2/20 = 0.1


def test_sphere_behaviour_at_large_learning_rates_is_not_divergence_below_one():
    """원문과 다른 점: x²+y² 에서 lr=0.5 는 한 번에 원점, lr=1.0 은 진동, lr>1 부터 발산한다."""
    one_step = run(opt.VanillaGD(0.5), opt.SPHERE)
    assert not one_step.diverged
    assert opt.iterations_to_reach(one_step.path, 1e-12) == 1

    bounce = run(opt.VanillaGD(1.0), opt.SPHERE)
    assert not bounce.diverged
    assert np.allclose(
        opt.radii(bounce.path), np.hypot(5, 5)
    )  # 반지름이 변하지 않고 (5,5) ↔ (−5,−5)

    assert run(opt.VanillaGD(1.1), opt.SPHERE).diverged


def test_stability_limit_of_ellipse_is_lr_0_1():
    assert not run(opt.VanillaGD(0.09), opt.ELLIPSE).diverged
    assert opt.radii(run(opt.VanillaGD(0.09), opt.ELLIPSE).path)[-1] < 1e-6
    assert run(opt.VanillaGD(0.11), opt.ELLIPSE).diverged


def test_diverged_path_stops_and_stays_finite():
    result = run(opt.VanillaGD(1.0), opt.ELLIPSE)
    assert result.diverged
    assert len(result.path) < 101
    assert np.all(np.isfinite(result.path))


# --- Momentum 효과 (실측 결과를 그대로 고정) ----------------------------------
def iterations(optimizer, objective, n_iter, radius=1e-3):
    """반경 radius 에 처음 들어가는 반복 횟수 (못 들어가면 무한대)."""
    reached = opt.iterations_to_reach(run(optimizer, objective, n_iter).path, radius)
    return np.inf if reached is None else reached


def test_momentum_wins_when_learning_rate_is_small():
    """x²+10y², lr=0.01: GD 는 300회에도 못 들어가고 Momentum 은 들어간다 (Momentum 의 이점)."""
    gd = iterations(opt.VanillaGD(0.01), opt.ELLIPSE, 300)
    mo = iterations(opt.Momentum(0.01, 0.9), opt.ELLIPSE, 300)
    assert gd == np.inf
    assert mo < 300


def test_momentum_is_not_faster_at_the_mission_settings():
    """원문과 다른 점: 미션이 지정한 lr 에서는 β=0.9 Momentum 이 더 느리다 (감쇠율 √β ≈ 0.949 > GD)."""
    for objective, lr in ((opt.SPHERE, 0.1), (opt.ELLIPSE, 0.05), (opt.ELLIPSE, 0.09)):
        gd = iterations(opt.VanillaGD(lr), objective, 100)
        mo = iterations(opt.Momentum(lr, 0.9), objective, 100)
        assert gd < mo


def test_momentum_still_converges_at_mission_settings():
    for objective, lr in ((opt.SPHERE, 0.1), (opt.ELLIPSE, 0.05)):
        final = opt.radii(run(opt.Momentum(lr, 0.9), objective, 100).path)[-1]
        assert final < 0.1  # 느리지만 반경 0.1 안에는 들어온다


# --- 그래프 --------------------------------------------------------------------
def test_gd_convergence_figure(tmp_path):
    fig, result = opt.plot_gd_convergence(save_path=tmp_path / "gd.png")
    assert len(fig.axes) == 2
    assert opt.radii(result.path)[-1] < 0.1
    assert (tmp_path / "gd.png").stat().st_size > 0


def test_lr_divergence_figure(tmp_path):
    fig = opt.plot_lr_divergence(save_path=tmp_path / "lr.png")
    assert len(fig.axes) == 4
    assert (tmp_path / "lr.png").stat().st_size > 0


def test_momentum_vs_gd_figure_overlays_both_paths_in_each_panel(tmp_path):
    fig, results = opt.plot_momentum_vs_gd(save_path=tmp_path / "cmp.png")
    assert len(fig.axes) == 4
    assert len(results) == 4
    for ax in fig.axes:
        assert len(ax.get_legend().get_texts()) == 2  # 한 축에 GD 와 Momentum 두 경로
    assert (tmp_path / "cmp.png").stat().st_size > 0


def test_lr_divergence_legend_labels_each_regime_correctly(tmp_path):
    """lr=1.1 (x²+y²) 은 30회 안에 임계값 1e6 에 닿지 않아도 발산으로 표시해야 한다."""
    fig = opt.plot_lr_divergence()
    for ax in (fig.axes[0], fig.axes[2]):  # x²+y² 의 경로 축과 거리 축
        labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert "lr=0.1: converges" in labels
        assert "lr=0.5: origin in 1 step" in labels
        assert "lr=1.0: bounces, no convergence" in labels
        assert "lr=1.1: diverges" in labels
    labels = {t.get_text() for t in fig.axes[1].get_legend().get_texts()}
    assert {
        "lr=0.05: converges",
        "lr=0.09: converges",
        "lr=0.5: diverges",
        "lr=1.0: diverges",
    } <= labels
