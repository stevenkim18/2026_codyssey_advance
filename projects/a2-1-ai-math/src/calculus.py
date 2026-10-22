"""미적분 모듈 (A2-1): 중심차분 수치 미분과 Gradient 시각화.

구성
    1. 수치 미분: 중심차분 f'(x) ≈ [f(x+h) − f(x−h)] / 2h 를 1변수와 다변수로 구현한다.
    2. Gradient 시각화: f(x, y) = x² + y² 의 등고선 위에 ∇f 를 화살표로 그려
       Gradient 가 등고선에 수직임을 확인한다.

함수 f 는 좌표를 첫 번째 축에 담은 배열 p (p[0] = x, p[1] = y) 를 받는다.
그래서 점 하나 (2,) 에도, 격자 (2, H, W) 에도 같은 f 를 그대로 쓸 수 있다.

실행: `uv run python -m src.calculus` (프로젝트 폴더에서) 하면 그래프가 `outputs/`에 저장된다.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_H = 1e-5


def paraboloid(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x² + y². 등고선은 원점 중심의 원이고 ∇f = (2x, 2y) 이다."""
    return p[0] ** 2 + p[1] ** 2


def numerical_derivative(
    f: Callable[[float], float], x: float, h: float = DEFAULT_H
) -> float:
    """중심차분으로 1변수 함수의 도함수 값을 근사한다.

    f'(x) ≈ [f(x + h) − f(x − h)] / (2h)

    테일러 전개에서 f(x ± h) = f ± h f' + h²/2 f'' ± h³/6 f''' + … 이므로 h² 항 이하가 상쇄되어
    오차가 O(h²) 이다 (전진차분은 O(h)). h 가 너무 작으면 반올림 오차가 커지므로 1e-5 안팎을 쓴다.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def numerical_gradient(
    f: Callable[[np.ndarray], np.ndarray], point: np.ndarray, h: float = DEFAULT_H
) -> np.ndarray:
    """중심차분으로 다변수 함수의 Gradient ∇f = (∂f/∂x₁, …, ∂f/∂xₙ) 를 근사한다.

    ∂f/∂xᵢ ≈ [f(p + h eᵢ) − f(p − h eᵢ)] / (2h)

    point 의 첫 번째 축이 좌표이다. (n,) 이면 한 점의 기울기를, (n, ...) 이면 모든 점의 기울기를
    한 번에 (n, ...) 로 돌려준다.
    """
    p = np.asarray(point, dtype=float)
    grad = np.zeros_like(p)
    for i in range(p.shape[0]):
        step = np.zeros_like(p)
        step[i] = h
        grad[i] = (f(p + step) - f(p - step)) / (2.0 * h)
    return grad


class DerivativeCheck(NamedTuple):
    """수치 미분과 해석적 미분의 비교 결과."""

    numerical: float
    analytic: float
    abs_error: float
    passed: bool


def check_derivative(
    f: Callable[[float], float],
    analytic_derivative: Callable[[float], float],
    x: float,
    tol: float = 1e-4,
    h: float = DEFAULT_H,
) -> DerivativeCheck:
    """x 에서 수치 미분과 해석적 미분의 절대 오차가 tol 이내인지 확인한다."""
    numerical = numerical_derivative(f, x, h)
    analytic = analytic_derivative(x)
    error = abs(numerical - analytic)
    return DerivativeCheck(
        float(numerical), float(analytic), float(error), bool(error <= tol)
    )


def perpendicularity(
    contour: "plt.contour.QuadContourSet", f: Callable[[np.ndarray], np.ndarray]
) -> float:
    """그려진 등고선의 접선과 Gradient 가 얼마나 수직인지 최대 |cos| 로 잰다.

    등고선을 이루는 선분마다 접선 t = vᵢ₊₁ − vᵢ 와 선분 중점에서의 수치 Gradient g 를 구해
    |t·g| / (‖t‖‖g‖) 를 계산한다. 수직이면 0 이다. 값이 작을수록(≈ 0) 수직에 가깝다.
    """
    worst = 0.0
    for path in contour.get_paths():
        for polygon in path.to_polygons(closed_only=False):
            tangent = np.diff(polygon, axis=0)
            mid = (polygon[:-1] + polygon[1:]) / 2.0
            for t, m in zip(tangent, mid, strict=True):
                g = numerical_gradient(f, m)
                denom = np.linalg.norm(t) * np.linalg.norm(g)
                if denom > 0:
                    worst = max(worst, abs(float(t @ g)) / float(denom))
    return worst


def plot_gradient_field(
    f: Callable[[np.ndarray], np.ndarray] = paraboloid,
    radii: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    n_arrows: int = 8,
    lim: float = 5.0,
    save_path: Path | str | None = None,
) -> tuple[Figure, float]:
    """f 의 등고선 위 여러 점에서 Gradient 화살표를 그리고, 수직 정도(최대 |cos|)를 함께 돌려준다.

    화살표(빨강)는 수치 Gradient ∇f 로, 값이 가장 빠르게 커지는 방향이다.
    짧은 점선(검정)은 그 점에서 등고선의 접선이고, 둘이 직각을 이룬다.
    f 가 x²+y² 이면 radii 의 각 반지름 r 이 등고선 f = r² 에 해당한다.
    """
    axis = np.linspace(-lim, lim, 400)
    x, y = np.meshgrid(axis, axis)
    z = f(np.stack([x, y]))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    levels = [r**2 for r in radii]
    contour = ax.contour(x, y, z, levels=levels, colors="0.55", linewidths=1)
    ax.clabel(contour, fmt=lambda v: f"f={v:g}", fontsize=8)

    angles = np.linspace(0, 2 * np.pi, n_arrows, endpoint=False) + np.pi / 8
    points = np.array(
        [[r * np.cos(a), r * np.sin(a)] for r in radii for a in angles]
    ).T  # (2, N)
    grad = numerical_gradient(f, points)

    ax.quiver(
        points[0],
        points[1],
        grad[0],
        grad[1],
        color="#d62728",
        angles="xy",
        scale_units="xy",
        scale=8,
        width=0.006,
        label=r"gradient $\nabla f$ (arrow length ∝ $\|\nabla f\|$)",
        zorder=4,
    )
    # 접선: ∇f 를 90° 돌린 방향 (−g_y, g_x) 의 단위 벡터. 등고선의 접선 방향이다.
    norm = np.linalg.norm(grad, axis=0)
    tangent = np.stack([-grad[1], grad[0]]) / norm
    for point, t in zip(points.T, tangent.T, strict=True):
        a, b = point - 0.35 * t, point + 0.35 * t
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", ls=":", lw=1.4, zorder=3)
    ax.plot([], [], color="black", ls=":", label="contour tangent")
    ax.scatter(points[0], points[1], s=10, color="black", zorder=5)

    worst = perpendicularity(contour, f)
    ax.set_title(
        f"Gradient is perpendicular to contours of f(x, y) = x² + y²\nmax |cos(tangent, ∇f)| = {worst:.1e}"
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, worst


def _save(fig: Figure, save_path: Path | str | None) -> None:
    """save_path 가 있으면 폴더를 만들고 PNG 로 저장한다."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    """수치 미분을 검증하고 Gradient 그래프를 outputs/ 에 저장한다."""
    np.random.seed(SEED)

    check = check_derivative(lambda x: x**2, lambda x: 2 * x, x=3.0)
    print("== f(x) = x², x = 3 에서 수치 미분 (허용 오차 1e-4) ==")
    print(
        f"numerical={check.numerical:.10f}  analytic={check.analytic:.1f}  error={check.abs_error:.2e}  {'OK' if check.passed else 'FAIL'}"
    )

    point = np.array([3.0, 4.0])
    print(
        f"\n∇f(3, 4) numerical={numerical_gradient(paraboloid, point)}  analytic={2 * point}"
    )

    _, worst = plot_gradient_field(save_path=OUTPUT_DIR / "gradient_contour.png")
    print(f"\n등고선 접선과 Gradient 의 최대 |cos| = {worst:.2e} (0 이면 정확히 수직)")
    plt.close("all")


if __name__ == "__main__":
    main()
