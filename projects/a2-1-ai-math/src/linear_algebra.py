"""선형대수: 기하 변환, Power Iteration, SVD 이미지 압축 (A2-1).

1. 단위 원에 회전 R(θ), 스케일링 S(sx, sy), 전단 Sh(k)를 적용하고 det(A)와 면적비를 비교한다.
2. Power Iteration으로 최대 고유값을 구하고 `np.linalg.eig` 결과와 비교한다.
3. SVD로 이미지를 rank-k 근사해 압축/복원한다.

`np.linalg.eig`는 `reference_dominant_eigenpair()` 한 곳에서 Power Iteration 검증용으로만 쓴다.

실행: 프로젝트 폴더(`projects/a2-1-ai-math`)에서 ``uv run python -m src.linear_algebra``
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook

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


# ----------------------------------------------------------------------
# 1. 기하 변환
# ----------------------------------------------------------------------
def rotation_matrix(theta: float) -> np.ndarray:
    r"""반시계 방향으로 ``theta``(라디안)만큼 회전시키는 2x2 행렬 R(θ).

    .. math:: R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}

    det R = cos²θ + sin²θ = 1 이므로 면적과 방향을 보존한다.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    r"""x축으로 ``sx``배, y축으로 ``sy``배 늘리는 대각 행렬 S(sx, sy).

    .. math:: S(s_x, s_y) = \begin{pmatrix}s_x & 0 \\ 0 & s_y\end{pmatrix}

    det S = sx·sy 이므로 면적이 sx·sy배가 된다. S(2, 0.5)는 면적을 보존한다.
    """
    return np.array([[sx, 0.0], [0.0, sy]])


def shear_matrix(k: float) -> np.ndarray:
    r"""x 좌표를 y에 비례해 밀어내는 전단(shear) 행렬 Sh(k).

    .. math:: Sh(k) = \begin{pmatrix}1 & k \\ 0 & 1\end{pmatrix}

    det Sh = 1 이므로 모양은 기울어져도 면적은 변하지 않는다.
    """
    return np.array([[1.0, k], [0.0, 1.0]])


def unit_circle(n_points: int = 360) -> np.ndarray:
    """단위 원 위의 점 ``n_points``개를 (2, n_points) 배열로 돌려준다."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.vstack([np.cos(angles), np.sin(angles)])


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(2, n) 점 집합에 2x2 행렬을 곱해 변환된 점을 돌려준다. 점마다 ``A @ p``다."""
    return np.asarray(matrix) @ np.asarray(points)


def polygon_area(points: np.ndarray) -> float:
    r"""신발끈 공식으로 (2, n) 꼭짓점 다각형의 면적을 구한다.

    .. math:: \text{Area} = \tfrac12\left|\sum_i (x_i y_{i+1} - x_{i+1} y_i)\right|
    """
    x, y = points
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def check_area_ratio(matrix: np.ndarray, n_points: int = 360) -> dict[str, float]:
    r"""변환 전후 면적비가 ``|det(A)|``와 맞는지 확인한다.

    선형 변환은 모든 도형의 면적을 ``|det A|``배로 만든다. 단위 원을 다각형으로
    근사해 면적을 직접 재고 ``area_after / area_before``를 ``|det A|``와 비교한다.

    Returns:
        det, area_before, area_after, ratio, rel_error(= |ratio - |det|| / |det|)를 담은 dict.
        det = 0이면 나눌 수 없으므로 rel_error는 절대 오차 |ratio|를 쓴다.
    """
    circle = unit_circle(n_points)
    before = polygon_area(circle)
    after = polygon_area(apply_transform(matrix, circle))
    det = float(np.linalg.det(matrix))
    ratio = after / before
    scale = abs(det) if det != 0.0 else 1.0
    return {
        "det": det,
        "area_before": before,
        "area_after": after,
        "ratio": ratio,
        "rel_error": abs(ratio - abs(det)) / scale,
    }


def default_transforms() -> dict[str, np.ndarray]:
    """미션의 변환 R(45°), S(2, 0.5), Sh(1)에 det가 1이 아닌 경우와 합성 변환을 더한 세트.

    R, S(2, 0.5), Sh(1)은 모두 det = 1이라 면적비가 1로만 나온다. det ≠ 1인 S(1.5, 2)(det 3)와
    det = 0인 특이 행렬을 함께 넣어 행렬식이 면적 배율이라는 점을 확인한다. R·S와 S·R은
    행렬 곱이 "변환을 차례로 적용"이며 순서를 바꾸면 결과가 달라진다는 것을 보여 준다.
    """
    rot = rotation_matrix(np.deg2rad(45.0))
    scale = scaling_matrix(2.0, 0.5)
    shear = shear_matrix(1.0)
    return {
        "R(45°) 회전": rot,
        "S(2, 0.5) 스케일링": scale,
        "Sh(1) 전단": shear,
        "S(1.5, 2) 스케일링 (det≠1)": scaling_matrix(1.5, 2.0),
        "R·S (스케일링 후 회전)": rot @ scale,
        "S·R (회전 후 스케일링)": scale @ rot,
        "특이 행렬 (det=0)": np.array([[1.0, 2.0], [2.0, 4.0]]),
    }


def plot_transformations(
    transforms: dict[str, np.ndarray] | None = None,
    n_points: int = 360,
    filename: str | None = "linear_transformations.png",
) -> plt.Figure:
    """단위 원의 변환 전후를 패널별로 그리고 마지막 패널에 det와 면적비 표를 둔다.

    점선은 변환 전 단위 원, 파란 영역은 변환 후 도형이다. 주황/청록 화살표는
    기저벡터 e₁, e₂가 옮겨 간 위치(= 행렬의 첫 번째, 두 번째 열)다.
    """
    setup_style()
    transforms = transforms if transforms is not None else default_transforms()
    circle = unit_circle(n_points)
    limit = 1.2 * max(
        np.abs(apply_transform(a, circle)).max() for a in transforms.values()
    )

    ncols = 4
    nrows = int(np.ceil((len(transforms) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, matrix) in zip(axes, transforms.items()):
        check = check_area_ratio(matrix, n_points)
        out = apply_transform(matrix, circle)
        ax.plot(*circle, color=INK_SECONDARY, linestyle="--", linewidth=1.3)
        ax.fill(*out, color=BLUE, alpha=0.16)
        ax.plot(*out, color=BLUE)
        for column, color in enumerate((ORANGE, AQUA)):
            ax.annotate(
                "",
                xy=matrix[:, column],
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0},
            )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.set_title(name, loc="left")
        ax.text(
            0.03,
            0.04,
            f"det = {check['det']:+.3f}\n면적비 = {check['ratio']:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            color=INK_SECONDARY,
        )

    summary = axes[len(transforms)]
    summary.axis("off")
    lines = ["변환별 det(A)와 면적비 (오차 < 1% 기준)", ""]
    for name, matrix in transforms.items():
        check = check_area_ratio(matrix, n_points)
        mark = "통과" if check["rel_error"] < 0.01 else "실패"
        lines.append(f"{name}")
        lines.append(
            f"   det {check['det']:+.3f} · 면적비 {check['ratio']:.4f} · 오차 {check['rel_error']:.0e} {mark}"
        )
    lines += ["", "점선: 변환 전 단위 원", "주황 화살표: A·e₁   청록 화살표: A·e₂"]
    summary.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9, color=INK_SECONDARY)

    for ax in axes[len(transforms) + 1 :]:
        ax.axis("off")
    fig.suptitle("단위 원에 선형 변환을 적용한 결과", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if filename:
        save_figure(fig, filename)
    return fig


# ----------------------------------------------------------------------
# 2. Power Iteration
# ----------------------------------------------------------------------
def power_iteration(
    matrix: np.ndarray,
    num_iters: int = 1000,
    tol: float = 1e-12,
    seed: int = SEED,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Power Iteration으로 절댓값이 가장 큰 고유값과 고유벡터를 구한다.

    .. math::
        v_{t+1} = \frac{A v_t}{\lVert A v_t \rVert}, \qquad
        \lambda_t = v_t^\top A v_t \;(\lVert v_t \rVert = 1)

    A의 고유벡터 x₁, ..., xₙ으로 v₀ = Σ cᵢxᵢ 라 쓰면 Aᵗv₀ = Σ cᵢλᵢᵗxᵢ 다.
    |λ₁| > |λ₂| 이면 λ₁ᵗ항이 지배해 v가 x₁으로 수렴하고, 오차는 |λ₂/λ₁|ᵗ 속도로 줄어든다.

    제약: 지배 고유값이 하나여야 한다. 절댓값이 같은 고유값이 둘(예: 회전 행렬의 복소수 쌍)
    이면 수렴하지 않는다. 시작 벡터가 x₁과 직교하면(c₁ = 0) 실패하므로 무작위 벡터로 시작한다.

    Args:
        matrix: 정사각 실수 행렬
        num_iters: 최대 반복 횟수
        tol: 연속한 고유값 추정치의 상대 변화가 이 값보다 작으면 멈춘다
        seed: 시작 벡터를 만드는 난수 시드

    Returns:
        (고유값 추정치, 단위 고유벡터, 반복마다의 고유값 추정치 배열)
    """
    a = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(a.shape[0])
    v /= np.linalg.norm(v)
    eigenvalue = float(v @ a @ v)
    history = [eigenvalue]
    for _ in range(num_iters):
        w = a @ v
        norm = np.linalg.norm(w)
        if norm == 0.0:  # Av = 0 이면 고유값 0
            eigenvalue = 0.0
            history.append(eigenvalue)
            break
        v = w / norm
        updated = float(v @ a @ v)
        history.append(updated)
        converged = abs(updated - eigenvalue) <= tol * max(1.0, abs(updated))
        eigenvalue = updated
        if converged:
            break
    return eigenvalue, v, np.array(history)


def reference_dominant_eigenpair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """`np.linalg.eig`로 절댓값이 가장 큰 고유값과 단위 고유벡터를 구한다 (검증 기준값).

    이 함수가 이 프로젝트에서 `np.linalg.eig`를 쓰는 유일한 곳이며,
    Power Iteration 결과를 검증할 때만 호출한다. 지배 고유값이 실수라고 가정한다.
    """
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    dominant = int(np.argmax(np.abs(values)))
    vector = vectors[:, dominant].real
    return float(values[dominant].real), vector / np.linalg.norm(vector)


def verify_power_iteration(
    matrix: np.ndarray, tol: float = 0.05, num_iters: int = 1000
) -> dict[str, float]:
    """Power Iteration 결과를 `np.linalg.eig`의 최대 고유값과 비교한다.

    Returns:
        power(Power Iteration 고유값), reference(eig의 지배 고유값), rel_error,
        cosine(두 고유벡터 사이 |cos|), iterations, passed(rel_error < tol)
    """
    a = np.asarray(matrix, dtype=float)
    power_value, power_vector, history = power_iteration(a, num_iters=num_iters)
    reference, reference_vector = reference_dominant_eigenpair(a)
    rel_error = abs(power_value - reference) / abs(reference)
    return {
        "power": power_value,
        "reference": reference,
        "rel_error": rel_error,
        "cosine": float(abs(power_vector @ reference_vector)),
        "iterations": len(history) - 1,
        "passed": bool(rel_error < tol),
    }


def power_iteration_examples(seed: int = SEED) -> dict[str, np.ndarray]:
    """검증에 쓰는 행렬 세 개: 비대칭 2x2, 대칭 3x3, 무작위 데이터의 공분산 5x5."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((200, 5)) * np.array([3.0, 2.0, 1.5, 1.0, 0.5])
    return {
        "비대칭 2x2 (λ=5, 2)": np.array([[4.0, 1.0], [2.0, 3.0]]),
        "대칭 3x3": np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]),
        "공분산 5x5": np.cov(data, rowvar=False),
    }


def plot_power_iteration(
    matrices: dict[str, np.ndarray] | None = None,
    filename: str | None = "power_iteration_convergence.png",
) -> plt.Figure:
    """반복 횟수에 따라 고유값 오차가 줄어드는 모습을 로그 축으로 그린다."""
    setup_style()
    matrices = matrices if matrices is not None else power_iteration_examples()
    colors = (BLUE, ORANGE, AQUA, YELLOW)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (name, matrix), color in zip(matrices.items(), colors):
        _, _, history = power_iteration(matrix, num_iters=60, tol=0.0)
        reference, _ = reference_dominant_eigenpair(matrix)
        error = np.maximum(np.abs(history - reference) / abs(reference), 1e-17)
        ax.semilogy(error, color=color, label=name)
    ax.axhline(0.05, color=INK_SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(
        0.99,
        0.05,
        "허용 오차 5%",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("반복 횟수")
    ax.set_ylabel("최대 고유값의 상대 오차 (로그)")
    ax.set_title(
        "Power Iteration 수렴: 반복마다 오차가 일정 비율로 줄어든다", loc="left"
    )
    ax.legend()
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


# ----------------------------------------------------------------------
# 3. SVD 이미지 압축
# ----------------------------------------------------------------------
def load_sample_image() -> np.ndarray:
    """matplotlib에 들어 있는 사진(grace_hopper.jpg)을 0~1 그레이스케일 (600, 512)로 읽는다."""
    rgb = plt.imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))
    rgb = rgb.astype(float) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114])


def svd_decompose(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""이미지 행렬 A (m×n)를 A = U Σ Vᵀ 로 분해한다 (경제형: U는 m×r, Vᵀ는 r×n)."""
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    return u, s, vt


def svd_reconstruct(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int) -> np.ndarray:
    r"""특이값 상위 ``k``개만 남긴 rank-k 근사 Aₖ를 만든다.

    .. math:: A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top = U_k \Sigma_k V_k^\top

    Eckart-Young 정리에 따라 Aₖ는 rank가 k 이하인 행렬 중 A에 프로베니우스 노름으로
    가장 가깝다.
    """
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def svd_compress(image: np.ndarray, k: int) -> np.ndarray:
    """``image``를 rank-``k``로 근사해 돌려준다 (화면 표시용으로 0~1에 자른다)."""
    u, s, vt = svd_decompose(image)
    return np.clip(svd_reconstruct(u, s, vt, k), 0.0, 1.0)


def storage_ratio(shape: tuple[int, int], k: int) -> float:
    """rank-k로 저장할 때 필요한 값의 개수 / 원본 값의 개수.

    U_k(m×k), σ(k), V_k(n×k)를 저장하므로 ``k(m + n + 1) / (m n)``이다.
    """
    m, n = shape
    return k * (m + n + 1) / (m * n)


def relative_error(image: np.ndarray, approx: np.ndarray) -> float:
    r"""상대 프로베니우스 오차 :math:`\lVert A - A_k\rVert_F / \lVert A\rVert_F`."""
    return float(np.linalg.norm(image - approx) / np.linalg.norm(image))


def psnr(image: np.ndarray, approx: np.ndarray, peak: float = 1.0) -> float:
    r"""PSNR(dB) = 10 log₁₀(peak² / MSE). 클수록 원본에 가깝다."""
    mse = float(np.mean((image - approx) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(peak**2 / mse)


def energy_ratio(s: np.ndarray, k: int) -> float:
    r"""상위 k개 특이값이 가진 에너지 비율 :math:`\sum_{i\le k}\sigma_i^2 / \sum_i\sigma_i^2`."""
    return float(np.sum(s[:k] ** 2) / np.sum(s**2))


def plot_svd_comparison(
    image: np.ndarray | None = None,
    ks: Sequence[int] = (10, 50, 100),
    filename: str | None = "svd_compression.png",
) -> plt.Figure:
    """원본과 ``k``별 복원 이미지를 나란히 놓고 저장 비율, 오차, PSNR을 제목에 적는다."""
    setup_style()
    image = load_sample_image() if image is None else image
    u, s, vt = svd_decompose(image)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3.6 * (len(ks) + 1), 4.6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"원본 ({image.shape[0]}×{image.shape[1]})")
    for ax, k in zip(axes[1:], ks):
        approx = np.clip(svd_reconstruct(u, s, vt, k), 0.0, 1.0)
        ax.imshow(approx, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"k = {k}\n저장 {storage_ratio(image.shape, k):.1%} · 오차 {relative_error(image, approx):.1%}"
            f" · {psnr(image, approx):.1f} dB",
            fontsize=10,
        )
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SVD rank-k 근사로 복원한 이미지", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if filename:
        save_figure(fig, filename)
    return fig


def plot_singular_values(
    image: np.ndarray | None = None,
    ks: Sequence[int] = (10, 50, 100),
    filename: str | None = "svd_singular_values.png",
) -> plt.Figure:
    """특이값 크기(로그)와 누적 에너지 비율을 그려 왜 소수의 특이값으로 충분한지 보여 준다."""
    setup_style()
    image = load_sample_image() if image is None else image
    _, s, _ = svd_decompose(image)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.semilogy(np.arange(1, len(s) + 1), s, color=BLUE)
    left.set_xlabel("순번 i")
    left.set_ylabel("특이값 σᵢ (로그)")
    left.set_title("특이값은 빠르게 작아진다", loc="left")
    cumulative = np.cumsum(s**2) / np.sum(s**2)
    right.plot(np.arange(1, len(s) + 1), cumulative, color=ORANGE)
    for i, k in enumerate(ks):  # 라벨이 겹치지 않게 아래로 계단식 배치
        right.scatter([k], [cumulative[k - 1]], color=ORANGE, s=36, zorder=3)
        right.annotate(
            f"k={k}: {cumulative[k - 1]:.1%}",
            (k, cumulative[k - 1]),
            textcoords="offset points",
            xytext=(24, -22 - 20 * i),
            fontsize=9,
            color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
        )
    right.set_xlabel("남긴 특이값 개수 k")
    right.set_ylabel("누적 에너지 비율")
    right.set_ylim(0.7, 1.01)
    right.set_title("상위 k개가 에너지 대부분을 가진다", loc="left")
    fig.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def main() -> None:
    """미션의 선형대수 요구사항을 실행하고 결과 PNG를 ``outputs/``에 저장한다."""
    np.random.seed(SEED)

    print("[1] 기하 변환: det(A)와 면적비 (허용 오차 1%)")
    for name, matrix in default_transforms().items():
        check = check_area_ratio(matrix)
        status = "통과" if check["rel_error"] < 0.01 else "실패"
        print(
            f"  {name:<24} det={check['det']:+.4f}  면적비={check['ratio']:.4f}"
            f"  오차={check['rel_error']:.2e}  {status}"
        )
    plot_transformations()

    print("\n[2] Power Iteration vs np.linalg.eig (허용 오차 5%)")
    for name, matrix in power_iteration_examples().items():
        result = verify_power_iteration(matrix)
        status = "통과" if result["passed"] else "실패"
        print(
            f"  {name:<20} power={result['power']:.6f}  eig={result['reference']:.6f}"
            f"  오차={result['rel_error']:.2e}  반복={result['iterations']}  {status}"
        )
    plot_power_iteration()

    print("\n[3] SVD 이미지 압축")
    image = load_sample_image()
    _, s, _ = svd_decompose(image)
    for k in (10, 50, 100):
        approx = svd_compress(image, k)
        print(
            f"  k={k:<4} 저장 {storage_ratio(image.shape, k):6.1%}  에너지 {energy_ratio(s, k):6.2%}"
            f"  오차 {relative_error(image, approx):6.2%}  PSNR {psnr(image, approx):5.1f} dB"
        )
    plot_svd_comparison(image)
    plot_singular_values(image)
    plt.close("all")
    print("\n그래프를 outputs/ 에 저장했습니다.")


if __name__ == "__main__":
    main()
