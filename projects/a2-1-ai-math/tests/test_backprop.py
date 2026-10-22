import numpy as np
import pytest

from src import backprop as bp


@pytest.fixture
def setup():
    x, params = bp.fixed_inputs()
    return x, params


def test_forward_values_match_hand_calculation_to_four_decimals(setup):
    x, params = setup
    cache = bp.forward(x, params)
    assert np.round(cache["z1"], 4).tolist() == [0.1, 0.3]
    assert np.round(cache["a1"], 4).tolist() == [0.5250, 0.5744]
    assert round(float(cache["z2"]), 4) == 0.6072
    assert round(float(cache["y_pred"]), 4) == 0.6473


def test_loss_matches_hand_calculation(setup):
    x, params = setup
    assert round(bp.loss_value(x, params, 1.0), 4) == 0.4350
    assert round(bp.loss_value(x, params, 0.0), 4) == 1.0421


def test_gradients_for_target_one_match_hand_calculation(setup):
    x, params = setup
    grads = bp.backward(bp.forward(x, params), params, 1.0)
    assert round(float(grads["dz2"]), 4) == -0.3527
    assert np.round(grads["dW2"], 4).tolist() == [-0.1852, -0.2026]
    assert round(float(grads["db2"]), 4) == -0.3527
    assert np.round(grads["da1"], 4).tolist() == [-0.1764, -0.2116]
    assert np.round(grads["dz1"], 4).tolist() == [-0.0440, -0.0517]
    assert np.round(grads["dW1"], 4).tolist() == [[-0.0440, 0.0], [-0.0517, 0.0]]
    assert np.round(grads["db1"], 4).tolist() == [-0.0440, -0.0517]


def test_gradients_for_target_zero_flip_sign_of_output_error(setup):
    x, params = setup
    grads = bp.backward(bp.forward(x, params), params, 0.0)
    assert round(float(grads["dz2"]), 4) == 0.6473
    assert np.round(grads["dW2"], 4).tolist() == [0.3398, 0.3718]


def test_shapes_of_every_quantity(setup):
    x, params = setup
    cache = bp.forward(x, params)
    grads = bp.backward(cache, params, 1.0)
    assert bp.shapes(cache) == {
        "x": (2,),
        "z1": (2,),
        "a1": (2,),
        "z2": (),
        "y_pred": (),
    }
    assert bp.shapes(grads) == {
        "dz2": (),
        "dW2": (2,),
        "db2": (),
        "da1": (2,),
        "dz1": (2,),
        "dW1": (2, 2),
        "db1": (2,),
        "dx": (2,),
    }
    for name in ("W1", "b1", "W2", "b2"):
        assert (
            grads["d" + name].shape == params[name].shape
        )  # 기울기는 파라미터와 같은 shape


@pytest.mark.parametrize("y_true", [0.0, 1.0])
def test_backprop_matches_numerical_gradient(setup, y_true):
    x, params = setup
    errors = bp.gradient_check(x, params, y_true)
    assert max(errors.values()) < 1e-8


def test_gradient_check_with_random_inputs_and_unfixed_parameters():
    rng = np.random.RandomState(42)
    x = rng.randn(2)
    params = {
        "W1": rng.randn(2, 2),
        "b1": rng.randn(2),
        "W2": rng.randn(2),
        "b2": np.array(rng.randn()),
    }
    assert max(bp.gradient_check(x, params, 1.0).values()) < 1e-7


def test_numerical_gradients_restore_the_parameters(setup):
    x, params = setup
    before = {k: v.copy() for k, v in params.items()}
    bp.numerical_gradients(x, params, 1.0)
    for name in params:
        assert np.array_equal(params[name], before[name])


def test_sigmoid_is_stable_for_extreme_inputs():
    with np.errstate(
        over="raise", invalid="raise", divide="raise"
    ):  # exp 언더플로(→0)는 무해하다
        out = bp.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_bce_clips_probabilities_so_loss_stays_finite():
    assert np.isfinite(bp.bce_loss(0.0, 1.0))
    assert np.isfinite(bp.bce_loss(1.0, 0.0))
