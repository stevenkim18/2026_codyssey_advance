"""미적분: 중심차분 수치 미분, Gradient, Hessian과 시각화 (A2-1).

1. 중심차분으로 f'(x)를 근사하고 해석적 정답과 비교한다 (f(x)=x², x=3 → 6).
2. 다변수 함수의 Gradient ∇f, Hessian H를 수치로 구한다.
3. f(x, y)=x²+y²의 등고선과 그에 수직인 Gradient 화살표를 그린다.

실행: 프로젝트 폴더(`projects/a2-1-ai-math`)에서 ``uv run python -m src.calculus``
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 스크립트로 직접 실행하면 프로젝트 폴더를 import 경로에 추가한다.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import (
    BLUE,
    INK_SECONDARY,
    ORANGE,
    SEED,
    save_figure,
    setup_style,
)

ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[np.ndarray], float]


# ----------------------------------------------------------------------
# 1. 수치 미분
# ----------------------------------------------------------------------
def central_difference(f: ScalarFunction, x: float, h: float = 1e-5) -> float:
    r"""중심차분 :math:`f'(x) \approx \dfrac{f(x+h) - f(x-h)}{2h}`.

    f(x±h)를 Taylor 전개하면 ``f(x+h) - f(x-h) = 2h f'(x) + (h³/3) f'''(x) + ...`` 이므로
    오차가 O(h²)이다. 전진차분 ``(f(x+h) - f(x))/h``의 오차 O(h)보다 훨씬 작다.
    h가 너무 작으면 반대로 부동소수점 반올림 오차(≈ ε/h)가 커지므로 1e-5 안팎이 적당하다.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def forward_difference(f: ScalarFunction, x: float, h: float = 1e-5) -> float:
    r"""전진차분 :math:`f'(x) \approx \dfrac{f(x+h) - f(x)}{h}` (비교용, 오차 O(h))."""
    return (f(x + h) - f(x)) / h


def verify_derivative(
    f: ScalarFunction,
    exact: float,
    x: float,
    h: float = 1e-5,
    tol: float = 1e-4,
) -> dict[str, float]:
    """수치 미분값이 해석적 정답 ``exact``와 ``tol`` 이내로 맞는지 확인한다.

    Returns:
        numeric, exact, abs_error, passed(abs_error < tol)를 담은 dict
    """
    numeric = central_difference(f, x, h)
    error = abs(numeric - exact)
    return {
        "numeric": numeric,
        "exact": exact,
        "abs_error": error,
        "passed": bool(error < tol),
    }


def numerical_gradient(
    f: VectorFunction, point: Sequence[float], h: float = 1e-5
) -> np.ndarray:
    r"""다변수 함수의 Gradient를 축마다 중심차분해 구한다.

    .. math:: \frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}
    """
    x = np.asarray(point, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2.0 * h)
    return grad


def numerical_hessian(
    f: VectorFunction, point: Sequence[float], h: float = 1e-4
) -> np.ndarray:
    r"""2차 편미분 행렬 Hessian을 중심차분으로 구한다 (Newton Method에 쓰인다).

    .. math::
        H_{ij} \approx \frac{f(x+he_i+he_j) - f(x+he_i-he_j) - f(x-he_i+he_j) + f(x-he_i-he_j)}{4h^2}
    """
    x = np.asarray(point, dtype=float)
    n = x.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h
            ej[j] = h
            hess[i, j] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h * h)
    return hess


# ----------------------------------------------------------------------
# 2. Gradient와 등고선의 수직 관계
# ----------------------------------------------------------------------
def paraboloid(point: Sequence[float]) -> float:
    """f(x, y) = x² + y²."""
    x, y = point
    return float(x * x + y * y)


def paraboloid_gradient(point: Sequence[float]) -> np.ndarray:
    """∇f = (2x, 2y). 항상 원점에서 바깥쪽을 가리킨다."""
    return 2.0 * np.asarray(point, dtype=float)


def contour_points(radii: Sequence[float], n_angles: int = 8) -> np.ndarray:
    """f = r²인 등고선(반지름 r인 원) 위의 점을 (N, 2) 배열로 만든다.

    각 반지름마다 ``n_angles``개의 점을 등간격으로 잡는다.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False) + np.pi / n_angles
    return np.array([[r * np.cos(a), r * np.sin(a)] for r in radii for a in angles])


def max_gradient_contour_cosine(
    f: VectorFunction = paraboloid,
    radii: Sequence[float] = (1.0, 2.0, 3.0),
    n_angles: int = 16,
) -> float:
    r"""등고선 접선과 수치 Gradient 사이 |cos|의 최댓값을 구한다.

    반지름 r인 원 위 점 ``p = r(cosθ, sinθ)``의 접선 방향은 ``(-sinθ, cosθ)``다.
    Gradient가 접선에 수직이면 단위 벡터끼리의 내적(= cos)이 0이어야 한다.
    """
    worst = 0.0
    for r in radii:
        for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
            p = r * np.array([np.cos(theta), np.sin(theta)])
            tangent = np.array([-np.sin(theta), np.cos(theta)])
            grad = numerical_gradient(f, p)
            cosine = abs(float(grad @ tangent)) / np.linalg.norm(grad)
            worst = max(worst, cosine)
    return worst


def plot_gradient_field(
    radii: Sequence[float] = (1.0, 2.0, 3.0),
    filename: str | None = "gradient_contour.png",
) -> plt.Figure:
    """f(x, y)=x²+y² 등고선 위에 접선(회색)과 수치 Gradient(주황 화살표)를 그린다.

    화살표는 데이터 좌표계 기준으로 그리고 가로세로 비율을 1:1로 맞춰서, 화면에서도
    Gradient가 등고선 접선과 직각으로 보인다. Gradient는 f가 가장 빠르게 커지는 방향이다.
    """
    setup_style()
    grid = np.linspace(-4.2, 4.2, 281)
    xx, yy = np.meshgrid(grid, grid)
    zz = xx**2 + yy**2

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    contours = ax.contour(
        xx, yy, zz, levels=[r * r for r in radii], colors=BLUE, linewidths=1.6
    )
    ax.clabel(contours, fmt=lambda v: f"f={v:.0f}", fontsize=9)

    points = contour_points(radii)
    grads = np.array([numerical_gradient(paraboloid, p) for p in points])
    angles = np.arctan2(points[:, 1], points[:, 0])
    tangents = np.stack([-np.sin(angles), np.cos(angles)], axis=1)
    for p, t in zip(points, tangents):  # 접선 방향 짧은 선분
        ax.plot(
            *np.stack([p - 0.35 * t, p + 0.35 * t], axis=1),
            color=INK_SECONDARY,
            linewidth=1.2,
        )
    ax.quiver(
        points[:, 0],
        points[:, 1],
        grads[:, 0],
        grads[:, 1],
        angles="xy",
        scale_units="xy",
        scale=6.0,
        color=ORANGE,
        width=0.006,
        zorder=3,
    )
    ax.scatter(*points.T, color=ORANGE, s=14, zorder=4)
    ax.scatter([0], [0], color=INK_SECONDARY, marker="x", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("f(x, y)=x²+y²: Gradient는 등고선에 수직", loc="left")
    ax.text(
        0.02,
        0.02,
        "주황 화살표: ∇f (길이 ÷ 6)\n회색 선분: 등고선 접선",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
    )
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


def plot_difference_error(
    filename: str | None = "numerical_derivative_error.png",
) -> plt.Figure:
    """간격 h에 따른 전진차분/중심차분 오차를 로그-로그로 비교한다 (f=eˣ, x=1).

    h를 줄이면 중심차분은 O(h²)로 빠르게 정확해지다가, h가 너무 작아지면
    반올림 오차 때문에 다시 커진다. 점선은 기본값 h=1e-5다.
    """
    setup_style()
    hs = np.logspace(-12, 0, 49)
    exact = np.e
    forward = np.array([abs(forward_difference(np.exp, 1.0, h) - exact) for h in hs])
    central = np.array([abs(central_difference(np.exp, 1.0, h) - exact) for h in hs])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(hs, forward, color=ORANGE, label="전진차분 (오차 O(h))")
    ax.loglog(hs, central, color=BLUE, label="중심차분 (오차 O(h²))")
    ax.axvline(1e-5, color=INK_SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(1.4e-5, 1e-1, "h = 1e-5", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("간격 h")
    ax.set_ylabel("|수치 미분 - 정답|")
    ax.set_title("수치 미분 오차: f(x)=eˣ, x=1", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def main() -> None:
    """미션의 미적분 요구사항을 실행하고 결과 PNG를 ``outputs/``에 저장한다."""
    np.random.seed(SEED)

    print("[1] 수치 미분: f(x)=x², x=3, 정답 6 (허용 오차 1e-4)")
    result = verify_derivative(lambda x: x * x, exact=6.0, x=3.0)
    forward = forward_difference(lambda x: x * x, 3.0)
    print(
        f"  중심차분 {result['numeric']:.10f}  오차 {result['abs_error']:.2e}"
        f"  {'통과' if result['passed'] else '실패'}   (참고: 전진차분 {forward:.6f}, 오차 {abs(forward - 6.0):.2e})"
    )

    print("\n[2] f(x,y)=x²+y² Gradient와 등고선 접선의 수직 여부")
    cosine = max_gradient_contour_cosine()
    print(f"  접선과 Gradient 사이 |cos| 최댓값 = {cosine:.2e} (0이면 수직)")
    point = np.array([1.0, 2.0])
    print(
        f"  (1, 2)에서 수치 ∇f = {numerical_gradient(paraboloid, point)}, 정답 {paraboloid_gradient(point)}"
    )
    print(f"  수치 Hessian =\n{np.round(numerical_hessian(paraboloid, point), 6)}")

    plot_gradient_field()
    plot_difference_error()
    plt.close("all")
    print("\n그래프를 outputs/ 에 저장했습니다.")


if __name__ == "__main__":
    main()
