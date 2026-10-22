"""경사하강법 계열 옵티마이저와 수렴 경로 시각화 (A2-1).

- `VanillaGD`: x ← x - lr·∇f
- `Momentum`: v ← β v + ∇f,  x ← x - lr·v
- `Adam` (보너스): 1차/2차 모멘트 추정 + 편향 보정
- `Newton` (보너스): x ← x - H⁻¹∇f

모든 옵티마이저는 ``step(x, grad) -> 새 x`` 형태로 같은 방식으로 쓴다. 상태(속도, 모멘트)는
객체가 들고 있으므로 새 실험마다 새 객체를 만들거나 ``reset()``한다.

실행: 프로젝트 폴더(`projects/a2-1-ai-math`)에서 ``uv run python -m src.optimizer``
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# 스크립트로 직접 실행하면 프로젝트 폴더를 import 경로에 추가한다.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    ORANGE,
    SEED,
    YELLOW,
    save_figure,
    setup_style,
)

START = (4.0, 3.0)
CONVERGED_RADIUS = 1e-3
BLOW_UP_RADIUS = 1e8


# ----------------------------------------------------------------------
# 목적 함수
# ----------------------------------------------------------------------
class Objective(NamedTuple):
    """최적화 대상 함수와 그 Gradient, Hessian (모두 2차원 입력)."""

    name: str
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Callable[[np.ndarray], np.ndarray]


def _sphere_f(p: np.ndarray) -> float:
    """f(x, y) = x² + y²."""
    return float(p[0] ** 2 + p[1] ** 2)


def _ill_f(p: np.ndarray) -> float:
    """f(x, y) = x² + 10y². Hessian 고윳값이 2와 20이라 조건수가 10인 길쭉한 골짜기다."""
    return float(p[0] ** 2 + 10.0 * p[1] ** 2)


def _rosenbrock_f(p: np.ndarray) -> float:
    """Rosenbrock f(x, y) = (1-x)² + 100(y-x²)². 비2차 함수이고 최솟값은 (1, 1)이다."""
    return float((1.0 - p[0]) ** 2 + 100.0 * (p[1] - p[0] ** 2) ** 2)


def _rosenbrock_grad(p: np.ndarray) -> np.ndarray:
    """Rosenbrock의 Gradient."""
    x, y = p
    return np.array([-2.0 * (1.0 - x) - 400.0 * x * (y - x * x), 200.0 * (y - x * x)])


def _rosenbrock_hess(p: np.ndarray) -> np.ndarray:
    """Rosenbrock의 Hessian."""
    x, y = p
    return np.array(
        [[2.0 - 400.0 * (y - 3.0 * x * x), -400.0 * x], [-400.0 * x, 200.0]]
    )


SPHERE = Objective(
    "f(x,y)=x²+y²",
    _sphere_f,
    lambda p: 2.0 * np.asarray(p, dtype=float),
    lambda p: np.diag([2.0, 2.0]),
)
ILL_CONDITIONED = Objective(
    "f(x,y)=x²+10y²",
    _ill_f,
    lambda p: np.array([2.0 * p[0], 20.0 * p[1]]),
    lambda p: np.diag([2.0, 20.0]),
)
ROSENBROCK = Objective("Rosenbrock", _rosenbrock_f, _rosenbrock_grad, _rosenbrock_hess)


# ----------------------------------------------------------------------
# 옵티마이저
# ----------------------------------------------------------------------
class Optimizer:
    """옵티마이저 공통 인터페이스."""

    name = "optimizer"

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """현재 위치 ``x``와 Gradient로 다음 위치를 돌려준다."""
        raise NotImplementedError

    def reset(self) -> None:
        """내부 상태(속도, 모멘트, 스텝 수)를 초기화한다."""


class VanillaGD(Optimizer):
    r"""기본 경사하강법 :math:`x_{t+1} = x_t - \eta \nabla f(x_t)`.

    f=x²+y²이면 ``x ← (1 - 2η)x``이므로 |1 - 2η| < 1, 즉 0 < η < 1일 때만 수렴한다.
    η = 0.5이면 한 번에 0에 도달하고, η = 1이면 ±로 진동만 하며, η > 1이면 발산한다.
    일반적으로 Hessian의 최대 고윳값이 L일 때 η < 2/L이어야 안정하다.
    """

    name = "Vanilla GD"

    def __init__(self, lr: float = 0.1):
        """``lr``: 학습률 η."""
        self.lr = lr

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """``x - lr * grad``를 돌려준다."""
        return x - self.lr * grad


class Momentum(Optimizer):
    r"""모멘텀 경사하강법 (heavy ball).

    .. math:: v_{t+1} = \beta v_t + \nabla f(x_t), \qquad x_{t+1} = x_t - \eta v_{t+1}

    Gradient가 일정한 방향이면 속도가 쌓여 유효 학습률이 ``η/(1-β)``까지 커지고,
    부호가 번갈아 바뀌는 방향(골짜기의 좁은 축)에서는 서로 상쇄된다.
    """

    name = "Momentum"

    def __init__(self, lr: float = 0.1, beta: float = 0.9):
        """``lr``: 학습률 η, ``beta``: 이전 속도를 유지하는 비율 β."""
        self.lr = lr
        self.beta = beta
        self.velocity: np.ndarray | None = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """속도를 ``β v + grad``로 갱신하고 ``x - lr * v``를 돌려준다."""
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.beta * self.velocity + grad
        return x - self.lr * self.velocity

    def reset(self) -> None:
        """속도를 비운다."""
        self.velocity = None


class Adam(Optimizer):
    r"""Adam (보너스): Gradient의 1차·2차 모멘트로 좌표마다 스텝 크기를 조절한다.

    .. math::
        m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
        s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2

        \hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat s_t = \frac{s_t}{1-\beta_2^t}, \quad
        x_{t+1} = x_t - \eta\,\frac{\hat m_t}{\sqrt{\hat s_t} + \varepsilon}

    m, s를 0에서 시작하므로 초반에 0쪽으로 치우친다. ``1-β^t``로 나눠 이를 보정한다.
    Gradient의 크기가 아니라 부호와 일관성으로 움직이므로 스텝은 대략 ``η`` 이하다.
    """

    name = "Adam"

    def __init__(
        self,
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """``lr``: 학습률 η, ``beta1``/``beta2``: 1차/2차 모멘트의 감쇠율, ``eps``: 0 나누기 방지."""
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.reset()

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """모멘트를 갱신하고 편향 보정한 Adam 스텝으로 이동한 위치를 돌려준다."""
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(x)
            self.s = np.zeros_like(x)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1**self.t)
        s_hat = self.s / (1.0 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)

    def reset(self) -> None:
        """스텝 수와 1차/2차 모멘트를 초기화한다."""
        self.t = 0
        self.m: np.ndarray | None = None
        self.s: np.ndarray | None = None


class Newton(Optimizer):
    r"""Newton Method (보너스): 2차 Taylor 근사의 최솟값으로 한 번에 이동한다.

    .. math:: f(x+d) \approx f(x) + \nabla f^\top d + \tfrac12 d^\top H d
        \;\Rightarrow\; d = -H^{-1}\nabla f, \qquad x_{t+1} = x_t - H^{-1}\nabla f

    f가 2차 함수이면 Taylor 근사가 정확하므로 단 한 번에 최솟값에 도달한다. 대신
    Hessian 계산과 선형 시스템 풀이(차원 n에서 O(n³))가 필요하고, H가 양의 정부호가
    아니면(안장점 근처) 올라가는 방향으로 갈 수 있다. ``step_size < 1``이면 감쇠 Newton이다.
    """

    name = "Newton"

    def __init__(
        self, hessian: Callable[[np.ndarray], np.ndarray], step_size: float = 1.0
    ):
        """``hessian``: 위치를 받아 Hessian 행렬을 돌려주는 함수, ``step_size``: 감쇠 계수(1이면 순수 Newton)."""
        self.hessian = hessian
        self.step_size = step_size

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """``H d = grad``를 풀어 ``x - step_size * d``를 돌려준다."""
        direction = np.linalg.solve(self.hessian(x), grad)  # H⁻¹g를 역행렬 없이 푼다
        return x - self.step_size * direction


# ----------------------------------------------------------------------
# 실행과 분석
# ----------------------------------------------------------------------
class Trajectory(NamedTuple):
    """최적화 경로. ``path[0]``은 시작점이다."""

    path: np.ndarray  # (T+1, 2)
    values: np.ndarray  # (T+1,)
    diverged: bool

    @property
    def radius(self) -> np.ndarray:
        """각 단계에서 원점까지의 거리 ‖x‖ (두 목적 함수 모두 최솟값이 원점이다)."""
        return np.linalg.norm(self.path, axis=1)


def minimize(
    optimizer: Optimizer,
    objective: Objective,
    start: Sequence[float] = START,
    n_iters: int = 100,
    blow_up: float = BLOW_UP_RADIUS,
) -> Trajectory:
    """``n_iters``번 갱신하며 경로를 기록한다. ‖x‖가 ``blow_up``을 넘으면 발산으로 보고 멈춘다."""
    optimizer.reset()
    x = np.asarray(start, dtype=float)
    path = [x.copy()]
    diverged = False
    with np.errstate(all="ignore"):
        for _ in range(n_iters):
            x = optimizer.step(x, objective.grad(x))
            path.append(x.copy())
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > blow_up:
                diverged = True
                break
    path_array = np.array(path)
    with np.errstate(all="ignore"):
        values = np.array([objective.f(p) for p in path_array])
    return Trajectory(path_array, values, diverged)


def iterations_to_converge(
    make_optimizer: Callable[[], Optimizer],
    objective: Objective,
    start: Sequence[float] = START,
    tol: float = CONVERGED_RADIUS,
    max_iters: int = 5000,
) -> float:
    """원점까지 거리가 ``tol`` 아래로 처음 내려가는 반복 횟수. 못 가면 ``nan``."""
    trajectory = minimize(make_optimizer(), objective, start, max_iters)
    reached = np.where(trajectory.radius < tol)[0]
    return float(reached[0]) if len(reached) else float("nan")


def lr_grid() -> np.ndarray:
    """학습률 민감도 실험에 쓰는 로그 간격 격자 (0.002 ~ 2)."""
    return np.logspace(np.log10(0.002), np.log10(2.0), 31)


def sweep_iterations(
    factory: Callable[[float], Optimizer],
    objective: Objective = ILL_CONDITIONED,
    lrs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """학습률마다 수렴까지 걸린 반복 횟수를 구한다. 수렴하지 못하면 ``nan``."""
    lrs = lr_grid() if lrs is None else lrs
    counts = np.array(
        [iterations_to_converge(lambda lr=lr: factory(lr), objective) for lr in lrs]
    )
    return lrs, counts


def best_lr(lrs: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """반복 횟수가 가장 적은 (학습률, 횟수)."""
    index = int(np.nanargmin(counts))
    return float(lrs[index]), float(counts[index])


# ----------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------
def _contour(
    ax: plt.Axes,
    objective: Objective,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """``objective``의 등고선과 최솟값(원점)을 ``ax``에 그린다."""
    gx = np.linspace(*xlim, 200)
    gy = np.linspace(*ylim, 200)
    xx, yy = np.meshgrid(gx, gy)
    zz = np.vectorize(lambda a, b: objective.f(np.array([a, b])))(xx, yy)
    ax.contour(
        xx,
        yy,
        zz,
        levels=np.geomspace(0.5, zz.max(), 9),
        colors=INK_SECONDARY,
        linewidths=0.8,
        alpha=0.55,
    )
    ax.scatter([0], [0], marker="x", color=INK_SECONDARY, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def _plot_path(
    ax: plt.Axes,
    trajectory: Trajectory,
    color: str,
    label: str,
    max_points: int | None = None,
) -> None:
    """경로의 처음 ``max_points``번(없으면 전체)을 점이 찍힌 선으로 그린다."""
    path = trajectory.path if max_points is None else trajectory.path[: max_points + 1]
    ax.plot(
        path[:, 0],
        path[:, 1],
        marker="o",
        markersize=3,
        color=color,
        linewidth=1.4,
        label=label,
    )


def plot_gd_convergence(filename: str | None = "gd_convergence.png") -> plt.Figure:
    """Vanilla GD(lr=0.1)가 f=x²+y²의 최솟값으로 가는 경로와 100회 후 반경."""
    setup_style()
    trajectory = minimize(VanillaGD(lr=0.1), SPHERE, START, 100)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    _contour(left, SPHERE, (-5, 5), (-4, 4))
    _plot_path(left, trajectory, BLUE, "경로", max_points=30)
    left.set_aspect("equal")
    left.set_title("경로 (처음 30회): x ← 0.8x 로 직선으로 접근", loc="left")
    right.semilogy(np.maximum(trajectory.radius, 1e-16), color=BLUE)
    right.axhline(0.1, color=ORANGE, linestyle="--", linewidth=1.2)
    right.text(99, 0.13, "목표 반경 0.1", ha="right", fontsize=9, color=INK_SECONDARY)
    final = trajectory.radius[-1]
    right.annotate(
        f"100회 후 반경 {final:.1e}",
        (100, final),
        textcoords="offset points",
        xytext=(-10, 0),
        ha="right",
        va="center",
        fontsize=9,
        color=INK_SECONDARY,
    )
    right.set_xlabel("반복 횟수")
    right.set_ylabel("원점까지 거리 ‖x‖ (로그)")
    right.set_title("Vanilla GD, lr=0.1, f=x²+y², 시작 (4, 3)", loc="left")
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


def plot_lr_sweep(filename: str | None = "lr_divergence.png") -> plt.Figure:
    """학습률별 ‖x‖ 변화를 GD(왼쪽)와 Momentum β=0.9(오른쪽)로 비교한다.

    위 행은 f=x²+y², 아래 행은 f=x²+10y²다. 100회까지 그리며, 발산하면 선이 중간에 끝난다(‖x‖ > 1e8).
    """
    setup_style()
    settings = [
        (SPHERE, (0.1, 0.5, 1.0, 1.1)),
        (ILL_CONDITIONED, (0.05, 0.1, 0.5, 1.0)),
    ]
    colors = (BLUE, ORANGE, AQUA, YELLOW)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for row, (objective, lrs) in enumerate(settings):
        for col, (title, factory) in enumerate(
            (
                ("Vanilla GD", lambda lr: VanillaGD(lr)),
                ("Momentum β=0.9", lambda lr: Momentum(lr, 0.9)),
            )
        ):
            ax = axes[row, col]
            for lr, color in zip(lrs, colors):
                trajectory = minimize(factory(lr), objective, START, 100)
                radius = np.maximum(trajectory.radius, 1e-16)
                ax.semilogy(radius, color=color, label=f"lr={lr}")
                if trajectory.diverged:
                    ax.scatter(
                        [len(radius) - 1],
                        [radius[-1]],
                        color=color,
                        marker="X",
                        s=50,
                        zorder=4,
                    )
            ax.axhline(0.1, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_ylim(1e-18, 1e14)  # 위쪽 여백에 범례를 둔다
            ax.set_title(f"{objective.name} · {title}", loc="left")
            ax.legend(loc="upper right", ncol=2, fontsize=9)
            if row == 1:
                ax.set_xlabel("반복 횟수")
            if col == 0:
                ax.set_ylabel("‖x‖ (로그)")
    fig.suptitle(
        "학습률과 발산: X 표시는 발산(‖x‖ > 1e8)으로 중단된 지점",
        x=0.01,
        ha="left",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if filename:
        save_figure(fig, filename)
    return fig


def plot_momentum_vs_gd(
    lr: float = 0.02, filename: str | None = "momentum_vs_gd.png"
) -> plt.Figure:
    """f=x²+10y²에서 같은 학습률의 GD와 Momentum(β=0.9)을 비교하고 학습률 민감도를 그린다.

    왼쪽은 경로, 가운데는 ‖x‖, 오른쪽은 학습률별 수렴(‖x‖ < 1e-3)까지 걸린 반복 횟수다.
    """
    setup_style()
    gd = minimize(VanillaGD(lr), ILL_CONDITIONED, START, 200)
    mom = minimize(Momentum(lr, 0.9), ILL_CONDITIONED, START, 200)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    _contour(axes[0], ILL_CONDITIONED, (-1, 5), (-3.5, 3.5))
    _plot_path(axes[0], gd, BLUE, "Vanilla GD", 60)
    _plot_path(axes[0], mom, ORANGE, "Momentum", 60)
    axes[0].set_title(f"경로 (처음 60회), lr={lr}", loc="left")
    axes[0].legend(loc="upper right")

    axes[1].semilogy(np.maximum(gd.radius, 1e-16), color=BLUE, label="Vanilla GD")
    axes[1].semilogy(np.maximum(mom.radius, 1e-16), color=ORANGE, label="Momentum")
    axes[1].axhline(CONVERGED_RADIUS, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("반복 횟수")
    axes[1].set_ylabel("‖x‖ (로그)")
    axes[1].set_title(f"수렴 속도, lr={lr}", loc="left")
    axes[1].legend()

    lrs, gd_counts = sweep_iterations(lambda r: VanillaGD(r))
    _, mom_counts = sweep_iterations(lambda r: Momentum(r, 0.9))
    axes[2].loglog(lrs, gd_counts, color=BLUE, label="Vanilla GD")
    axes[2].loglog(lrs, mom_counts, color=ORANGE, label="Momentum")
    for limit, text, y, va in (
        (0.1, "GD 안정 한계\nlr < 2/20 = 0.1", 0.96, "top"),
        (0.19, "Momentum 안정 한계\nlr < 2(1+β)/20 = 0.19", 0.04, "bottom"),
    ):
        axes[2].axvline(limit, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
        axes[2].text(
            limit * 0.97,
            y,
            text,
            transform=axes[2].get_xaxis_transform(),
            ha="right",
            va=va,
            fontsize=9,
            color=INK_SECONDARY,
        )
    axes[2].set_xlabel("학습률 lr")
    axes[2].set_ylabel("수렴까지 반복 횟수 (로그)")
    axes[2].set_title("학습률 민감도 (선이 끊기면 수렴 실패)", loc="left")
    axes[2].legend(loc="lower left")
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


def plot_optimizer_comparison(
    lr: float = 0.05, filename: str | None = "optimizer_comparison.png"
) -> plt.Figure:
    """보너스: GD, Momentum, Adam, Newton을 f=x²+10y²에서 비교한다.

    왼쪽과 가운데는 세 옵티마이저에 같은 학습률 ``lr``을 줬을 때의 경로와 ‖x‖다.
    오른쪽은 학습률마다 수렴(‖x‖ < 1e-3)까지 걸린 횟수로, 각자 최적 학습률에서의 속도와
    학습률에 얼마나 민감한지를 보여 준다. Newton은 학습률이 없고 Hessian을 쓴다.
    """
    setup_style()
    candidates = {
        "Vanilla GD": (lambda r: VanillaGD(r), BLUE),
        "Momentum": (lambda r: Momentum(r, 0.9), ORANGE),
        "Adam": (lambda r: Adam(r), AQUA),
    }
    runs: dict[str, Trajectory] = {
        name: minimize(factory(lr), ILL_CONDITIONED, START, 250)
        for name, (factory, _) in candidates.items()
    }
    newton = minimize(Newton(ILL_CONDITIONED.hess), ILL_CONDITIONED, START, 5)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    _contour(axes[0], ILL_CONDITIONED, (-1, 5), (-3.5, 3.5))
    for name, trajectory in runs.items():
        _plot_path(axes[0], trajectory, candidates[name][1], name, 40)
    _plot_path(axes[0], newton, YELLOW, "Newton", 1)
    axes[0].set_title(f"경로 (처음 40회), 같은 lr={lr}", loc="left")
    axes[0].legend(loc="lower right", fontsize=9)

    for name, trajectory in runs.items():
        reached = np.where(trajectory.radius < CONVERGED_RADIUS)[0]
        suffix = f"{reached[0]}회" if len(reached) else "미수렴"
        axes[1].semilogy(
            np.maximum(trajectory.radius, 1e-16),
            color=candidates[name][1],
            label=f"{name} ({suffix})",
        )
    axes[1].scatter(
        [1],
        [1e-8],
        marker="v",
        color=YELLOW,
        s=70,
        zorder=4,
        label="Newton (1회에 ‖x‖=0)",
    )
    axes[1].axhline(CONVERGED_RADIUS, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
    axes[1].set_ylim(1e-8, 1e1)
    axes[1].set_xlabel("반복 횟수")
    axes[1].set_ylabel("‖x‖ (로그)")
    axes[1].set_title(f"수렴 속도, 같은 lr={lr} (괄호: ‖x‖<1e-3 도달)", loc="left")
    axes[1].legend(fontsize=9)

    for name, (factory, color) in candidates.items():
        lrs, counts = sweep_iterations(factory)
        best, count = best_lr(lrs, counts)
        axes[2].loglog(
            lrs,
            counts,
            color=color,
            label=f"{name} (최적 lr={best:.2g}, {count:.0f}회)",
        )
    axes[2].axhline(1, color=YELLOW, linewidth=2.0, label="Newton (1회)")
    axes[2].set_xlabel("학습률 lr")
    axes[2].set_ylabel("수렴까지 반복 횟수 (로그)")
    axes[2].set_title("학습률 민감도 (선이 끊기면 수렴 실패)", loc="left")
    axes[2].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def _describe(trajectory: Trajectory) -> str:
    """경로의 끝 상태를 한 줄 문자열(발산 여부 또는 최종 반경)로 만든다."""
    if trajectory.diverged:
        return f"발산 ({len(trajectory.path) - 1}회 만에 ‖x‖ > 1e8)"
    return f"‖x‖ = {trajectory.radius[-1]:.3e}"


def main() -> None:
    """미션의 최적화 요구사항과 보너스를 실행하고 결과 PNG를 ``outputs/``에 저장한다."""
    np.random.seed(SEED)

    print("[1] Vanilla GD: f=x²+y², lr=0.1, 100회, 시작 (4, 3) → 반경 0.1 이내?")
    trajectory = minimize(VanillaGD(0.1), SPHERE, START, 100)
    print(
        f"  최종 반경 {trajectory.radius[-1]:.3e}  {'통과' if trajectory.radius[-1] < 0.1 else '실패'}"
    )

    print("\n[2] 학습률과 발산 (100회, 시작 (4, 3))")
    for objective, lrs in (
        (SPHERE, (0.1, 0.5, 1.0, 1.1)),
        (ILL_CONDITIONED, (0.05, 0.1, 0.5, 1.0)),
    ):
        print(f"  {objective.name}")
        for lr in lrs:
            gd = minimize(VanillaGD(lr), objective, START, 100)
            mom = minimize(Momentum(lr, 0.9), objective, START, 100)
            print(f"    lr={lr:<4}  GD: {_describe(gd):<34} Momentum: {_describe(mom)}")

    print("\n[3] f=x²+10y²: 수렴(‖x‖<1e-3)까지 반복 횟수")
    for lr in (0.01, 0.02, 0.05, 0.09):
        gd = iterations_to_converge(lambda lr=lr: VanillaGD(lr), ILL_CONDITIONED)
        mom = iterations_to_converge(lambda lr=lr: Momentum(lr, 0.9), ILL_CONDITIONED)
        print(f"  lr={lr:<5} GD {gd:>5.0f}회   Momentum {mom:>5.0f}회")
    plot_gd_convergence()
    plot_lr_sweep()
    plot_momentum_vs_gd()

    print("\n[4] 보너스: 학습률을 튜닝한 3종 + Newton (f=x²+10y², ‖x‖<1e-3)")
    for name, factory in (
        ("Vanilla GD", lambda lr: VanillaGD(lr)),
        ("Momentum β=0.9", lambda lr: Momentum(lr, 0.9)),
        ("Adam", lambda lr: Adam(lr)),
    ):
        lr, count = best_lr(*sweep_iterations(factory))
        print(f"  {name:<15} 최적 lr={lr:.3g}  {count:.0f}회")
    newton = minimize(Newton(ILL_CONDITIONED.hess), ILL_CONDITIONED, START, 5)
    print(f"  {'Newton':<15} 1회 후 ‖x‖ = {newton.radius[1]:.1e}")
    rosen = minimize(Newton(ROSENBROCK.hess), ROSENBROCK, (-1.2, 1.0), 50)
    reached = np.where(np.linalg.norm(rosen.path - 1.0, axis=1) < 1e-8)[0]
    print(
        f"  Rosenbrock(비2차)에서 Newton은 {reached[0]}회 만에 (1, 1)에 도달"
        if len(reached)
        else "  Rosenbrock 미수렴"
    )
    plot_optimizer_comparison()
    plt.close("all")
    print("\n그래프를 outputs/ 에 저장했습니다.")


if __name__ == "__main__":
    main()
