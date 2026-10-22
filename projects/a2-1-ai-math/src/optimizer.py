"""최적화 모듈 (A2-1): Vanilla GD, Momentum 옵티마이저와 수렴 경로 시각화.

구성
    1. 옵티마이저 클래스: `VanillaGD`, `Momentum`
       VanillaGD:  θ ← θ − η ∇f(θ)
       Momentum:   v ← β v + ∇f(θ),   θ ← θ − η v
    2. 테스트 함수: f(x, y) = x² + y²,  f(x, y) = x² + 10y²
    3. 시각화: 등고선 위 수렴 경로(점선), 학습률별 발산, Vanilla GD 와 Momentum 비교

이차함수 f = ½ θᵀHθ 에 GD 를 적용하면 H 의 고유값 λ 방향마다 θ ← (1 − ηλ) θ 로 줄어들므로
모든 λ 에 대해 |1 − ηλ| < 1, 즉 η < 2/λ_max 여야 수렴한다. x²+y² 는 λ = 2, 2 라서 η < 1,
x²+10y² 는 λ = 2, 20 이라서 η < 0.1 이다. 아래 그래프는 이 한계 근처의 거동을 보여 준다.

실행: `uv run python -m src.optimizer` (프로젝트 폴더에서) 하면 그래프가 `outputs/`에 저장된다.
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

START = (5.0, 5.0)
BLUE = "#1f77b4"
ORANGE = "#d55e00"


# ---------------------------------------------------------------------------
# 테스트 함수
# ---------------------------------------------------------------------------
def sphere(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x² + y². 최솟값은 원점이고 헤시안 고유값은 (2, 2) 이다."""
    return p[0] ** 2 + p[1] ** 2


def sphere_grad(p: np.ndarray) -> np.ndarray:
    """∇f = (2x, 2y)."""
    return 2.0 * p


def ellipse(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x² + 10y². y 방향이 10배 가파른 타원형 골짜기이고 헤시안 고유값은 (2, 20) 이다."""
    return p[0] ** 2 + 10.0 * p[1] ** 2


def ellipse_grad(p: np.ndarray) -> np.ndarray:
    """∇f = (2x, 20y)."""
    return np.array([2.0 * p[0], 20.0 * p[1]])


class Objective(NamedTuple):
    """최적화 대상 함수: 이름, 함수값 f(p), 기울기 ∇f(p)."""

    name: str
    f: Callable[[np.ndarray], np.ndarray]
    grad: Callable[[np.ndarray], np.ndarray]


SPHERE = Objective("f(x, y) = x² + y²", sphere, sphere_grad)
ELLIPSE = Objective("f(x, y) = x² + 10y²", ellipse, ellipse_grad)


# ---------------------------------------------------------------------------
# 옵티마이저
# ---------------------------------------------------------------------------
class Optimizer:
    """옵티마이저 공통 인터페이스. 하위 클래스는 `step` 을 구현한다.

    Args:
        lr: 학습률 η (양수)
    """

    def __init__(self, lr: float):
        """학습률을 저장한다. 양수가 아니면 ValueError."""
        if lr <= 0:
            raise ValueError(f"학습률은 양수여야 한다: {lr}")
        self.lr = lr

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """기울기로 파라미터를 한 번 갱신해 새 파라미터를 돌려준다 (입력은 바꾸지 않는다)."""
        raise NotImplementedError

    def reset(self) -> None:
        """내부 상태(속도 등)를 초기화한다. 상태가 없으면 아무 일도 하지 않는다."""


class VanillaGD(Optimizer):
    """가장 기본적인 경사하강법.

    θ ← θ − η ∇f(θ)
    """

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """θ − η·∇f 를 돌려준다."""
        return params - self.lr * grads


class Momentum(Optimizer):
    """속도(관성)를 누적하는 경사하강법.

    v ← β v + ∇f(θ)
    θ ← θ − η v

    기울기가 계속 같은 방향이면 속도가 최대 1/(1−β) 배까지 쌓여 빨라지고,
    부호가 번갈아 바뀌는 방향(지그재그)은 서로 상쇄되어 줄어든다.

    Args:
        lr: 학습률 η
        beta: 관성 계수 β (0 ≤ β < 1). 미션 기본값은 0.9
    """

    def __init__(self, lr: float, beta: float = 0.9):
        """학습률과 관성 계수를 저장하고 속도를 비워 둔다."""
        super().__init__(lr)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta 는 [0, 1) 이어야 한다: {beta}")
        self.beta = beta
        self.velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """속도를 갱신한 뒤 θ − η·v 를 돌려준다."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.beta * self.velocity + grads
        return params - self.lr * self.velocity

    def reset(self) -> None:
        """속도를 0 으로 되돌린다."""
        self.velocity = None


class Trajectory(NamedTuple):
    """최적화 경로. path[t] 는 t번 갱신한 뒤의 (x, y) 이고 path[0] 은 시작점이다."""

    path: np.ndarray
    diverged: bool


def minimize(
    optimizer: Optimizer,
    objective: Objective,
    start: tuple[float, float] = START,
    n_iter: int = 100,
    diverge_at: float = 1e6,
) -> Trajectory:
    """optimizer 로 n_iter 번 갱신하며 경로를 기록한다.

    원점으로부터의 거리가 diverge_at 을 넘거나 값이 유한하지 않으면 발산으로 보고 거기서 멈춘다.
    """
    optimizer.reset()
    p = np.asarray(start, dtype=float)
    path = [p]
    diverged = False
    for _ in range(n_iter):
        p = optimizer.step(p, objective.grad(p))
        path.append(p)
        if not np.all(np.isfinite(p)) or np.linalg.norm(p) > diverge_at:
            diverged = True
            break
    return Trajectory(np.array(path), diverged)


def radii(path: np.ndarray) -> np.ndarray:
    """경로의 각 점에서 원점(최솟값)까지의 거리 ‖θ‖ 를 돌려준다."""
    return np.linalg.norm(path, axis=1)


def iterations_to_reach(path: np.ndarray, radius: float) -> int | None:
    """원점 거리가 처음으로 radius 아래가 되는 갱신 횟수를 돌려준다. 끝까지 못 가면 None."""
    inside = np.flatnonzero(radii(path) < radius)
    return int(inside[0]) if inside.size else None


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def _draw_contour(ax: plt.Axes, objective: Objective, lim: float) -> None:
    """목적함수의 등고선을 회색으로 그린다."""
    axis = np.linspace(-lim, lim, 300)
    x, y = np.meshgrid(axis, axis)
    ax.contour(
        x, y, objective.f(np.stack([x, y])), levels=14, colors="0.8", linewidths=0.9
    )
    ax.plot(0, 0, marker="*", color="black", ms=11, zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_path(
    ax: plt.Axes, path: np.ndarray, color: str, label: str, max_markers: int = 40
) -> None:
    """경로를 점선과 점 표시로 그리고, 시작점(네모)과 끝점(×)을 표시한다."""
    every = max(1, len(path) // max_markers)
    ax.plot(
        path[:, 0],
        path[:, 1],
        ls=":",
        lw=1.4,
        marker="o",
        ms=3.5,
        markevery=every,
        color=color,
        label=label,
        zorder=4,
    )
    ax.plot(*path[0], marker="s", color=color, ms=6, zorder=5)
    ax.plot(*path[-1], marker="X", color=color, ms=9, mec="white", zorder=5)


def plot_gd_convergence(
    lr: float = 0.1,
    n_iter: int = 100,
    save_path: Path | str | None = None,
) -> tuple[Figure, Trajectory]:
    """f(x, y) = x² + y² 에서 Vanilla GD 의 수렴 경로(등고선 위 점선)와 원점 거리 변화를 그린다."""
    result = minimize(VanillaGD(lr), SPHERE, n_iter=n_iter)
    r = radii(result.path)
    reached = iterations_to_reach(result.path, 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _draw_contour(ax1, SPHERE, 6)
    _draw_path(ax1, result.path, BLUE, f"Vanilla GD (lr={lr})")
    ax1.set_title(
        f"{SPHERE.name}: start {START}, {n_iter} iterations\nfinal ({result.path[-1, 0]:.1e}, {result.path[-1, 1]:.1e})"
    )
    ax1.legend(loc="upper left", fontsize=8)

    ax2.semilogy(r, color=BLUE)
    ax2.axhline(0.1, color="0.4", ls="--", lw=1, label="radius 0.1")
    if reached is not None:
        ax2.axvline(reached, color="0.4", ls=":", lw=1)
        ax2.annotate(
            f"reaches radius 0.1\nafter {reached} iterations",
            (reached, 0.1),
            xytext=(reached + 8, 3),
            fontsize=9,
            arrowprops={"arrowstyle": "->"},
        )
    ax2.set_xlabel("iteration")
    ax2.set_ylabel(r"distance to origin $\|\theta\|$")
    ax2.set_title(f"final radius after {n_iter} iterations = {r[-1]:.1e}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, result


def _regime(trajectory: Trajectory) -> str:
    """경로의 거동을 한마디로 요약한다: 발산, 한 번에 원점, 진동(수렴 안 함), 수렴.

    발산은 임계값 플래그가 아니라 원점 거리 변화로 판단한다. 반복 횟수가 짧아 임계값에
    닿지 못해도, 거리가 시작 거리의 2배 이상으로 커졌다면 발산이다.
    """
    r = radii(trajectory.path)
    if trajectory.diverged or r[-1] >= 2.0 * r[0]:
        return "diverges"
    if r[-1] < 1e-12:
        return f"origin in {iterations_to_reach(trajectory.path, 1e-12)} step"
    if r[-1] > 0.5 * r[0]:
        return "bounces, no convergence"
    return "converges"


def plot_lr_divergence(
    sphere_lrs: tuple[float, ...] = (0.1, 0.5, 1.0, 1.1),
    ellipse_lrs: tuple[float, ...] = (0.05, 0.09, 0.5, 1.0),
    n_iter: int = 30,
    n_path: int = 8,
    save_path: Path | str | None = None,
) -> Figure:
    """학습률을 키워 가며 경로(위)와 원점 거리(아래)를 두 함수에서 비교해 발산을 보여 준다.

    경로는 처음 n_path 번만 그린다 (발산하면 그래프 밖으로 나가 버리기 때문이다).
    원점 거리는 로그 축이라 발산하면 오른쪽 위로 치솟는다.
    """
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for col, (objective, lrs) in enumerate(
        ((SPHERE, sphere_lrs), (ELLIPSE, ellipse_lrs))
    ):
        ax_path, ax_norm = axes[0, col], axes[1, col]
        _draw_contour(ax_path, objective, 6)
        for lr, color in zip(lrs, colors, strict=False):
            result = minimize(VanillaGD(lr), objective, n_iter=n_iter)
            label = f"lr={lr}: {_regime(result)}"
            _draw_path(ax_path, result.path[: n_path + 1], color, label)
            ax_norm.semilogy(
                radii(result.path), color=color, marker="o", ms=3, label=label
            )
        ax_norm.set_ylim(bottom=1e-3)
        limit = 2.0 / (2.0 if objective is SPHERE else 20.0)
        ax_path.set_title(
            f"{objective.name}: first {n_path} steps\nstable only for lr < 2/λmax = {limit:g}"
        )
        ax_path.legend(loc="upper left", fontsize=8)
        ax_norm.axhline(np.linalg.norm(START), color="0.5", ls="--", lw=1)
        ax_norm.set_xlabel("iteration")
        ax_norm.set_ylabel(r"$\|\theta\|$ (log scale)")
        ax_norm.set_title("distance to origin (dashed: start)")
        ax_norm.legend(fontsize=8)
        ax_norm.grid(alpha=0.3)
    fig.suptitle("Vanilla GD: effect of learning rate")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, save_path)
    return fig


class Comparison(NamedTuple):
    """한 설정에서 Vanilla GD 와 Momentum 을 비교한 결과."""

    objective: Objective
    lr: float
    n_iter: int
    gd: Trajectory
    momentum: Trajectory


def compare(
    objective: Objective, lr: float, n_iter: int, beta: float = 0.9
) -> Comparison:
    """같은 목적함수·학습률·시작점에서 Vanilla GD 와 Momentum(beta) 의 경로를 구한다."""
    gd = minimize(VanillaGD(lr), objective, n_iter=n_iter)
    mom = minimize(Momentum(lr, beta), objective, n_iter=n_iter)
    return Comparison(objective, lr, n_iter, gd, mom)


def comparison_settings() -> dict[str, tuple[Objective, float, int]]:
    """비교 그래프의 네 설정: (목적함수, 학습률, 반복 횟수). 앞의 둘은 미션이 지정한 설정이다."""
    return {
        "(a) mission: x²+y², lr=0.1": (SPHERE, 0.1, 100),
        "(b) mission: x²+10y², lr=0.05": (ELLIPSE, 0.05, 100),
        "(c) x²+10y², lr=0.09 (near stability limit)": (ELLIPSE, 0.09, 100),
        "(d) x²+10y², lr=0.01 (small lr)": (ELLIPSE, 0.01, 300),
    }


def plot_momentum_vs_gd(
    beta: float = 0.9,
    radius: float = 1e-3,
    save_path: Path | str | None = None,
) -> tuple[Figure, dict[str, Comparison]]:
    """Vanilla GD 와 Momentum(beta) 의 경로를 설정마다 한 축에 겹쳐 그린다.

    범례에는 원점 반경 radius 안에 처음 들어가는 반복 횟수를 적는다 (못 들어가면 'not reached').
    """
    results: dict[str, Comparison] = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for ax, (title, (objective, lr, n_iter)) in zip(
        axes.ravel(), comparison_settings().items(), strict=True
    ):
        comp = compare(objective, lr, n_iter, beta)
        results[title] = comp
        _draw_contour(ax, objective, 6)
        for traj, color, name in (
            (comp.gd, BLUE, "Vanilla GD"),
            (comp.momentum, ORANGE, f"Momentum (β={beta})"),
        ):
            reached = iterations_to_reach(traj.path, radius)
            if reached is not None:
                note = f"radius < {radius:g} at iteration {reached}"
            else:
                note = (
                    f"radius {radii(traj.path)[-1]:.1e} after {comp.n_iter} iterations"
                )
            _draw_path(ax, traj.path, color, f"{name}: {note}")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Vanilla GD vs Momentum, start (5, 5)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, save_path)
    return fig, results


def _save(fig: Figure, save_path: Path | str | None) -> None:
    """save_path 가 있으면 폴더를 만들고 PNG 로 저장한다."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    """그래프 세 장을 outputs/ 에 저장하고 비교 결과를 표로 출력한다."""
    np.random.seed(SEED)

    _, gd = plot_gd_convergence(save_path=OUTPUT_DIR / "gd_convergence.png")
    print("== Vanilla GD, f = x²+y², lr=0.1, 100 iterations ==")
    print(
        f"final = ({gd.path[-1, 0]:.2e}, {gd.path[-1, 1]:.2e}), radius = {radii(gd.path)[-1]:.2e}  {'OK' if radii(gd.path)[-1] < 0.1 else 'FAIL'}"
    )

    plot_lr_divergence(save_path=OUTPUT_DIR / "lr_divergence.png")
    print("\n== 학습률별 100회 후 원점 거리 ==")
    for objective, lrs in (
        (SPHERE, (0.1, 0.5, 1.0, 1.1)),
        (ELLIPSE, (0.05, 0.09, 0.1, 0.11, 0.5, 1.0)),
    ):
        for lr in lrs:
            t = minimize(VanillaGD(lr), objective, n_iter=100)
            state = "diverged" if t.diverged else f"radius {radii(t.path)[-1]:.3g}"
            print(f"{objective.name:20s} lr={lr:<5} {state}")

    _, results = plot_momentum_vs_gd(save_path=OUTPUT_DIR / "momentum_vs_gd.png")
    print(
        "\n== Vanilla GD vs Momentum(β=0.9): 반경 1e-3 도달 반복 횟수, 마지막 반경 =="
    )
    for title, comp in results.items():
        gd_n = iterations_to_reach(comp.gd.path, 1e-3)
        mo_n = iterations_to_reach(comp.momentum.path, 1e-3)
        print(
            f"{title:46s} GD: {gd_n!s:>11} (final {radii(comp.gd.path)[-1]:.1e})   "
            f"Momentum: {mo_n!s:>11} (final {radii(comp.momentum.path)[-1]:.1e})"
        )
    plt.close("all")


if __name__ == "__main__":
    main()
