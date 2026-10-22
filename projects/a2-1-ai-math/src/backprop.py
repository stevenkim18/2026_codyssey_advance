"""2층 신경망의 순전파/역전파를 NumPy로 구현하고 수치 미분으로 검증한다 (A2-1).

구조: 입력 2 → 은닉 2 (Sigmoid) → 출력 1 (Sigmoid), 손실은 이진 교차 엔트로피(BCE).

    z1 = W1 x + b1        a1 = σ(z1)
    z2 = W2 · a1 + b2     ŷ  = σ(z2)
    L  = -[y log ŷ + (1 - y) log(1 - ŷ)]

가중치 관례는 ``W @ x``다. ``W1[i, j]``는 입력 j에서 은닉 뉴런 i로 가는 가중치이고 shape은
(은닉, 입력)=(2, 2)다. ``W2``는 shape (2,)인 행 벡터, ``b2``는 스칼라다.

노트북 `notebooks/backprop_derivation.ipynb`가 이 모듈로 손계산 결과를 검증한다.
"""

from __future__ import annotations

import numpy as np

Params = dict[str, np.ndarray]


def sigmoid(z: np.ndarray) -> np.ndarray:
    r"""시그모이드 :math:`\sigma(z) = 1 / (1 + e^{-z})`.

    ``z``가 큰 음수여도 ``exp``가 넘치지 않도록 부호에 따라 식을 나눠 계산한다.
    """
    z = np.asarray(z, dtype=float)
    positive = z >= 0
    out = np.empty_like(z)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_loss(y_pred: float, y_true: float, eps: float = 1e-12) -> float:
    r"""이진 교차 엔트로피 :math:`-[y\log\hat y + (1-y)\log(1-\hat y)]`.

    ``log(0)``을 피하려고 ``ŷ``를 [eps, 1-eps]로 자른다.
    """
    p = float(np.clip(y_pred, eps, 1.0 - eps))
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def fixed_inputs() -> tuple[np.ndarray, Params]:
    """미션의 고정값 ``x``와 파라미터 ``W1, b1, W2, b2``를 돌려준다."""
    x = np.array([1.0, 0.0])
    params = {
        "W1": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "b1": np.array([0.0, 0.0]),
        "W2": np.array([0.5, 0.6]),
        "b2": np.array(0.0),
    }
    return x, params


def forward(x: np.ndarray, params: Params) -> dict[str, np.ndarray]:
    """순전파. 역전파에 필요한 중간값을 모두 담은 cache를 돌려준다.

    Returns:
        ``x, z1, a1, z2, y_pred``
    """
    z1 = params["W1"] @ x + params["b1"]
    a1 = sigmoid(z1)
    z2 = params["W2"] @ a1 + params["b2"]
    y_pred = sigmoid(z2)
    return {"x": x, "z1": z1, "a1": a1, "z2": z2, "y_pred": y_pred}


def backward(
    cache: dict[str, np.ndarray], params: Params, y_true: float
) -> dict[str, np.ndarray]:
    r"""연쇄 법칙으로 모든 기울기를 구한다 (출력 → 입력 순서).

    .. math::
        \frac{\partial L}{\partial z_2} = \hat y - y \qquad
        \frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2}\, a_1 \qquad
        \frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}

        \frac{\partial L}{\partial a_1} = W_2\,\frac{\partial L}{\partial z_2} \qquad
        \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1}\odot a_1(1-a_1) \qquad
        \frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1}\, x^\top \qquad
        \frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}

    출력층의 Sigmoid와 BCE를 합치면 ``σ'(z₂)``와 ``1/(ŷ(1-ŷ))``가 약분되어
    ``∂L/∂z₂ = ŷ - y``라는 간단한 식이 된다.

    Returns:
        ``dz2, dW2, db2, da1, dz1, dW1, db1, dx``
    """
    x, a1, y_pred = cache["x"], cache["a1"], cache["y_pred"]
    dz2 = y_pred - y_true
    dW2 = dz2 * a1
    db2 = dz2
    da1 = params["W2"] * dz2
    dz1 = da1 * a1 * (1.0 - a1)
    dW1 = np.outer(dz1, x)
    db1 = dz1
    dx = params["W1"].T @ dz1
    return {
        "dz2": np.asarray(dz2),
        "dW2": dW2,
        "db2": np.asarray(db2),
        "da1": da1,
        "dz1": dz1,
        "dW1": dW1,
        "db1": db1,
        "dx": dx,
    }


def loss_value(x: np.ndarray, params: Params, y_true: float) -> float:
    """순전파 후 BCE 손실을 계산한다."""
    return bce_loss(float(forward(x, params)["y_pred"]), y_true)


def numerical_gradients(
    x: np.ndarray, params: Params, y_true: float, h: float = 1e-6
) -> dict[str, np.ndarray]:
    """파라미터 ``W1, b1, W2, b2``의 기울기를 중심차분으로 구한다 (역전파 검증용).

    각 원소를 ±h 흔들어 ``(L(θ+h) - L(θ-h)) / 2h``를 계산한다.
    """
    grads: dict[str, np.ndarray] = {}
    for name, value in params.items():
        grad = np.zeros_like(value, dtype=float)
        for index in np.ndindex(value.shape):
            original = float(value[index])
            value[index] = original + h
            plus = loss_value(x, params, y_true)
            value[index] = original - h
            minus = loss_value(x, params, y_true)
            value[index] = original
            grad[index] = (plus - minus) / (2.0 * h)
        grads[name] = grad
    return grads


def gradient_check(
    x: np.ndarray, params: Params, y_true: float, h: float = 1e-6
) -> dict[str, float]:
    """역전파 기울기와 수치 기울기의 차이(절대 오차 최댓값)를 파라미터별로 돌려준다."""
    analytic = backward(forward(x, params), params, y_true)
    numeric = numerical_gradients(x, params, y_true, h)
    return {
        name: float(np.max(np.abs(analytic["d" + name] - numeric[name])))
        for name in params
    }


def shapes(values: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    """이름별 배열의 shape을 돌려준다. 스칼라는 ``()``다."""
    return {name: tuple(np.shape(v)) for name, v in values.items()}


def main() -> None:
    """고정값으로 순전파/역전파를 실행하고 값, shape, 수치 검증 결과를 출력한다."""
    np.random.seed(42)
    x, params = fixed_inputs()
    for y_true in (1.0, 0.0):
        cache = forward(x, params)
        grads = backward(cache, params, y_true)
        print(
            f"\n[y = {y_true:.0f}] L = {bce_loss(float(cache['y_pred']), y_true):.4f}"
        )
        for name, value in {**cache, **grads}.items():
            print(
                f"  {name:<7} shape={np.shape(value)!s:<8} {np.round(value, 4).tolist()}"
            )
        errors = gradient_check(x, params, y_true)
        print(
            "  수치 기울기와의 최대 오차:", {k: f"{v:.1e}" for k, v in errors.items()}
        )


if __name__ == "__main__":
    main()
