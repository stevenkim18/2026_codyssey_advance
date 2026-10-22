import numpy as np
import pytest

from src import probability as pr


# ---------------------------------------------------------------- 분포
def test_normal_pdf_known_values_and_normalization():
    assert pr.normal_pdf(0.0) == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))
    assert pr.normal_pdf(2.0, mu=2.0, sigma=0.5) == pytest.approx(
        1.0 / (0.5 * np.sqrt(2.0 * np.pi))
    )
    grid = np.linspace(-12, 12, 24001)
    for mu, sigma in ((0.0, 1.0), (2.0, 0.5)):
        assert np.sum(pr.normal_pdf(grid, mu, sigma)) * (
            grid[1] - grid[0]
        ) == pytest.approx(1.0, abs=1e-6)


def test_normal_pdf_is_symmetric_and_can_exceed_one():
    assert pr.normal_pdf(1.3, 0.0, 1.0) == pytest.approx(pr.normal_pdf(-1.3, 0.0, 1.0))
    assert pr.normal_pdf(0.0, 0.0, 0.1) > 1.0  # 밀도는 확률이 아니다


def test_bernoulli_pmf_sums_to_one_and_picks_the_right_probability():
    for p in (0.5, 0.8):
        pmf = pr.bernoulli_pmf(np.array([0, 1]), p)
        assert pmf.sum() == pytest.approx(1.0)
        assert pmf.tolist() == [pytest.approx(1 - p), pytest.approx(p)]


# ---------------------------------------------------------------- Softmax
def test_softmax_sums_to_one_within_1e_6_for_random_logits():
    rng = np.random.RandomState(42)
    logits = rng.randn(1000, 10) * 5.0
    assert np.max(np.abs(pr.softmax(logits).sum(axis=1) - 1.0)) < 1e-6


def test_softmax_is_shift_invariant_and_stable_for_large_logits():
    z = np.array([1.0, 2.0, 3.0])
    assert pr.softmax(z) == pytest.approx(pr.softmax(z + 1000.0))
    with np.errstate(all="raise"):
        big = pr.softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert big.sum() == pytest.approx(1.0, abs=1e-6)


def test_softmax_preserves_order_and_log_softmax_matches():
    z = np.array([0.5, -1.0, 2.0])
    assert np.argmax(pr.softmax(z)) == np.argmax(z)
    assert np.exp(pr.log_softmax(z)) == pytest.approx(pr.softmax(z))


# ---------------------------------------------------------------- MSE = 정규분포 MLE
def test_normal_nll_equals_affine_function_of_mse():
    rng = np.random.RandomState(42)
    y = rng.normal(3.0, 2.0, 50)
    mu, sigma = 2.5, 1.7
    expected = 0.5 * y.size * np.log(2 * np.pi * sigma**2) + y.size * pr.mse(y, mu) / (
        2 * sigma**2
    )
    assert pr.normal_nll(y, mu, sigma) == pytest.approx(expected)


def test_mean_minimizes_both_mse_and_normal_nll():
    rng = np.random.RandomState(42)
    y = rng.normal(3.0, 2.0, 200)
    mus = np.linspace(0.0, 6.0, 6001)
    nll_best = mus[np.argmin([pr.normal_nll(y, m, 2.0) for m in mus])]
    mse_best = mus[np.argmin([pr.mse(y, m) for m in mus])]
    assert nll_best == pytest.approx(mse_best)
    assert nll_best == pytest.approx(y.mean(), abs=1e-3)


def test_normal_mle_uses_divide_by_n_variance():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu, sigma = pr.normal_mle(y)
    assert mu == pytest.approx(2.5)
    assert sigma == pytest.approx(np.sqrt(1.25))  # 표본분산(n-1)이면 √(5/3)


# ---------------------------------------------------------------- Cross-Entropy = 베르누이/카테고리 MLE
def test_bernoulli_nll_is_n_times_bce_and_sample_mean_minimizes_it():
    y = np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    ps = np.linspace(0.01, 0.99, 981)
    assert pr.bernoulli_nll(y, np.full(y.size, 0.7)) == pytest.approx(
        y.size * pr.binary_cross_entropy(y, np.full(y.size, 0.7))
    )
    best = ps[np.argmin([pr.binary_cross_entropy(y, np.full(y.size, p)) for p in ps])]
    assert best == pytest.approx(pr.bernoulli_mle(y), abs=1e-3)


def test_binary_cross_entropy_hand_value_and_clipping():
    assert pr.binary_cross_entropy(np.array([1.0]), np.array([0.8])) == pytest.approx(
        -np.log(0.8)
    )
    assert np.isfinite(pr.binary_cross_entropy(np.array([1.0]), np.array([0.0])))


def test_categorical_cross_entropy_with_one_hot_is_negative_log_of_true_class_probability():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    onehot = np.array([[1, 0, 0], [0, 1, 0]])
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    assert pr.categorical_cross_entropy(onehot, probs) == pytest.approx(expected)
    assert pr.categorical_nll(np.array([0, 1]), probs) == pytest.approx(
        -(np.log(0.7) + np.log(0.8))
    )


def test_softmax_cross_entropy_gradient_is_p_minus_y_and_matches_numerical():
    logits = np.array([0.3, -1.2, 2.0])
    onehot = np.array([0.0, 0.0, 1.0])
    analytic = pr.softmax_cross_entropy_grad(logits, onehot)
    assert analytic == pytest.approx(pr.softmax(logits) - onehot)

    def loss(z):
        return pr.categorical_cross_entropy(onehot[None, :], pr.softmax(z)[None, :])

    numeric = np.array(
        [(loss(logits + 1e-6 * e) - loss(logits - 1e-6 * e)) / 2e-6 for e in np.eye(3)]
    )
    assert analytic == pytest.approx(numeric, abs=1e-8)


# ---------------------------------------------------------------- 정보 이론 (보너스)
def test_entropy_of_uniform_is_log_n_and_of_point_mass_is_zero():
    assert pr.entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))
    assert pr.entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_cross_entropy_equals_entropy_plus_kl():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    assert pr.cross_entropy(p, q) == pytest.approx(
        pr.entropy(p) + pr.kl_divergence(p, q)
    )
    assert pr.cross_entropy(p, p) == pytest.approx(pr.entropy(p))


def test_kl_is_nonnegative_zero_only_for_equal_and_asymmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    assert pr.kl_divergence(p, q) > 0.0
    assert pr.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert pr.kl_divergence(p, q) != pytest.approx(pr.kl_divergence(q, p))


def test_kl_with_one_hot_target_equals_cross_entropy():
    onehot = np.array([0.0, 1.0, 0.0])
    q = np.array([0.2, 0.5, 0.3])
    assert pr.kl_divergence(onehot, q) == pytest.approx(pr.cross_entropy(onehot, q))
    assert pr.cross_entropy(onehot, q) == pytest.approx(-np.log(0.5))


def test_gaussian_kl_matches_numerical_integration():
    grid = np.linspace(-15, 15, 300001)
    dx = grid[1] - grid[0]
    p = pr.normal_pdf(grid, 0.0, 1.0)
    q = pr.normal_pdf(grid, 1.0, 2.0)
    numeric = np.sum(p * np.log(p / q)) * dx
    assert pr.gaussian_kl(0.0, 1.0, 1.0, 2.0) == pytest.approx(numeric, abs=1e-6)
    assert pr.gaussian_kl(0.5, 1.5, 0.5, 1.5) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [np.array([0.5, 0.6]), np.array([-0.1, 1.1])])
def test_invalid_distributions_are_rejected(bad):
    with pytest.raises(ValueError):
        pr.entropy(bad)
    with pytest.raises(ValueError):
        pr.kl_divergence(np.array([0.5, 0.5]), bad)
