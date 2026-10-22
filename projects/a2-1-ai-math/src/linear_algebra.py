"""선형대수 모듈 (A2-1): 행렬 변환 시각화, Power Iteration, SVD 이미지 압축.

구성
    1. 행렬 변환: 회전 R(θ), 스케일링 S(sx, sy), 전단 Sh(k)를 단위 원과 격자에 적용하고
       det(A)가 면적 변화율과 같은지 확인한다.
    2. Power Iteration: v ← Av / ‖Av‖ 를 반복해 지배 고유값과 고유벡터를 구한다.
       `np.linalg.eig`는 `verify_*` 함수 안에서 결과를 검증할 때만 사용한다.
    3. SVD 압축: A = UΣVᵀ 에서 특이값 상위 k개만 남겨 이미지를 복원한다.

실행: `uv run python -m src.linear_algebra` (프로젝트 폴더에서) 하면 그래프가 `outputs/`에 저장된다.
"""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BLUE = "#1f77b4"
RED = "#d62728"


# ---------------------------------------------------------------------------
# 1. 행렬 변환
# ---------------------------------------------------------------------------
def rotation_matrix(theta: float) -> np.ndarray:
    """회전 행렬 R(θ)를 만든다. θ는 라디안이며 반시계 방향이 양수다.

    R(θ) = [[cos θ, -sin θ],
            [sin θ,  cos θ]]      det R = cos²θ + sin²θ = 1
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """스케일링 행렬 S(sx, sy)를 만든다. x축을 sx배, y축을 sy배 늘린다.

    S(sx, sy) = [[sx, 0],
                 [0, sy]]         det S = sx · sy
    """
    return np.array([[sx, 0.0], [0.0, sy]])


def shear_matrix(k: float) -> np.ndarray:
    """x 방향 전단 행렬 Sh(k)를 만든다. (x, y) → (x + k·y, y).

    Sh(k) = [[1, k],
             [0, 1]]              det Sh = 1 (밑변과 높이가 같아 면적이 보존된다)
    """
    return np.array([[1.0, k], [0.0, 1.0]])


def unit_circle(n_points: int = 100) -> np.ndarray:
    """단위 원 위의 점을 (2, n_points) 배열로 만든다.

    θ = linspace(0, 2π, n_points),  점 = (cos θ, sin θ)
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    return np.vstack([np.cos(theta), np.sin(theta)])


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(2, n) 점들에 2x2 행렬을 적용한다. p' = A p."""
    return matrix @ points


def polygon_area(points: np.ndarray) -> float:
    """신발끈 공식(shoelace)으로 다각형의 부호 있는 면적을 구한다.

    Area = ½ Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
    점이 반시계 방향이면 양수, 시계 방향(방향이 뒤집힌 경우)이면 음수다.
    """
    x, y = points
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class AreaCheck(NamedTuple):
    """행렬식과 면적비 비교 결과."""

    det: float
    area_ratio: float
    rel_error: float
    passed: bool


def check_area_ratio(
    matrix: np.ndarray, n_points: int = 100, tol: float = 0.01
) -> AreaCheck:
    """det(A)와 '변환 후 면적 / 변환 전 면적'을 비교해 오차가 tol 이내인지 확인한다.

    선형 변환은 모든 도형의 면적을 |det A|배로 바꾸고, 방향이 뒤집히면 부호가 바뀐다.
    그래서 부호 있는 면적비가 det(A)와 같아야 한다.
    오차는 |ratio − det| / |det| 이고, det ≈ 0 이면 분모 없이 절대 오차를 쓴다.
    """
    circle = unit_circle(n_points)
    area_before = polygon_area(circle)
    area_after = polygon_area(apply_transform(matrix, circle))
    ratio = area_after / area_before
    det = float(np.linalg.det(matrix))
    gap = abs(ratio - det)
    rel_error = gap / abs(det) if abs(det) > 1e-12 else gap
    return AreaCheck(det, float(ratio), float(rel_error), bool(rel_error <= tol))


def default_transforms(
    theta: float = np.pi / 4, k: float = 1.0
) -> dict[str, np.ndarray]:
    """미션의 세 변환(회전, S(2, 0.5), 전단)을 이름과 함께 돌려준다."""
    return {
        f"Rotation R({np.degrees(theta):.0f}°)": rotation_matrix(theta),
        "Scaling S(2, 0.5)": scaling_matrix(2.0, 0.5),
        f"Shear Sh({k:g})": shear_matrix(k),
    }


def _draw_transformation(
    ax: plt.Axes, matrix: np.ndarray, title: str, grid_extent: float = 1.5
) -> None:
    """한 축에 변환 전(파랑)과 후(빨강)의 단위 원과 격자를 겹쳐 그린다."""
    # 격자: np.meshgrid 로 만든 2D 점 집합을 행/열 방향 선으로 이어 그린다 (선형 변환은 직선을 직선으로 보낸다).
    lin = np.linspace(-grid_extent, grid_extent, 7)
    gx, gy = np.meshgrid(lin, lin)
    grid = np.stack([gx, gy])  # (2, 7, 7)
    moved = np.einsum("ij,jrc->irc", matrix, grid)

    for i in range(lin.size):
        ax.plot(grid[0, i], grid[1, i], color=BLUE, alpha=0.18, lw=0.8)
        ax.plot(grid[0, :, i], grid[1, :, i], color=BLUE, alpha=0.18, lw=0.8)
        ax.plot(moved[0, i], moved[1, i], color=RED, alpha=0.25, lw=0.8)
        ax.plot(moved[0, :, i], moved[1, :, i], color=RED, alpha=0.25, lw=0.8)

    circle = unit_circle()
    after = apply_transform(matrix, circle)
    ax.plot(*after, color=RED, lw=2, label="after: A · circle")
    ax.fill(*after, color=RED, alpha=0.08)
    ax.plot(*circle, color=BLUE, lw=2, ls="--", label="before: unit circle")

    # 회전은 원을 원으로 보내 전후 구분이 안 되므로, 기저 벡터 e1, e2 의 이동으로 변환을 보여 준다.
    basis = np.eye(2)
    for vectors, color in ((basis, BLUE), (matrix @ basis, RED)):
        ax.quiver(
            [0, 0],
            [0, 0],
            vectors[0],
            vectors[1],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.012,
            zorder=5,
        )

    check = check_area_ratio(matrix)
    ax.text(
        0.03,
        0.97,
        f"det(A) = {check.det:.3f}\narea ratio = {check.area_ratio:.3f}\nerror = {check.rel_error:.2%}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"},
    )

    limit = float(max(np.abs(moved).max(), np.abs(grid).max())) + 0.4
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_title(title)


def plot_transformations(
    transforms: dict[str, np.ndarray] | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """변환마다 한 축씩, 변환 전후를 겹쳐 그린 Figure를 만든다."""
    transforms = default_transforms() if transforms is None else transforms
    fig, axes = plt.subplots(
        1, len(transforms), figsize=(5 * len(transforms), 5), squeeze=False
    )
    for ax, (name, matrix) in zip(axes[0], transforms.items(), strict=True):
        _draw_transformation(ax, matrix, name)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Linear transformations of the unit circle, grid and basis vectors (arrows: e1, e2)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Power Iteration
# ---------------------------------------------------------------------------
class PowerIterationResult(NamedTuple):
    """Power Iteration 결과.

    Attributes:
        eigenvalue: 지배 고유값 (절댓값이 가장 큰 고유값)
        eigenvector: 단위 길이 고유벡터 (가장 큰 성분이 양수가 되도록 부호를 맞춘다)
        n_iter: 수행한 반복 횟수
        converged: tol 안에서 수렴했는지
        deltas: 반복마다의 방향 변화량 ‖vₖ − vₖ₋₁‖ (부호 무시)
    """

    eigenvalue: float
    eigenvector: np.ndarray
    n_iter: int
    converged: bool
    deltas: np.ndarray


def power_iteration(
    matrix: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: int = SEED,
) -> PowerIterationResult:
    """Power Iteration으로 절댓값이 가장 큰(지배) 고유값과 고유벡터를 구한다.

    알고리즘
        v₀ = 임의의 단위 벡터
        wₖ = A vₖ₋₁,   vₖ = wₖ / ‖wₖ‖
        λₖ = vₖᵀ A vₖ  (레일리 몫, ‖vₖ‖ = 1)
    A의 고유벡터를 eᵢ, 고유값을 λᵢ 라 하면 Aᵏ v₀ = Σ cᵢ λᵢᵏ eᵢ 에서 |λ₁| 항이 지배하므로
    vₖ 는 e₁ 방향으로 수렴하고, 오차는 |λ₂/λ₁|ᵏ 비율로 줄어든다.

    수렴 조건은 방향 변화량 min(‖vₖ − vₖ₋₁‖, ‖vₖ + vₖ₋₁‖) < tol 이다 (지배 고유값이
    음수이면 vₖ 의 부호가 매번 뒤집히므로 두 경우를 모두 본다).

    제한: 지배 고유값이 하나이고 실수여야 수렴한다. 회전 행렬처럼 복소 고유값이 지배하면
    converged=False 로 돌아온다.

    Args:
        matrix: 정방 행렬
        max_iter: 최대 반복 횟수
        tol: 방향 변화량 허용치
        seed: 초기 벡터 난수 시드

    Raises:
        ValueError: 정방 행렬이 아니거나, 반복 중 Av 가 영벡터가 되는 경우
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"정방 행렬이 필요하다: shape={a.shape}")

    v = np.random.default_rng(seed).normal(size=a.shape[0])
    v /= np.linalg.norm(v)

    deltas = []
    converged = False
    for _ in range(max_iter):
        w = a @ v
        norm = np.linalg.norm(w)
        if norm == 0.0:
            raise ValueError(
                "Av = 0 이다: 초기 벡터가 영공간에 있거나 행렬이 영행렬이다."
            )
        v_new = w / norm
        delta = min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v))
        deltas.append(delta)
        v = v_new
        if delta < tol:
            converged = True
            break

    eigenvalue = float(v @ a @ v)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return PowerIterationResult(eigenvalue, v, len(deltas), converged, np.array(deltas))


class EigenCheck(NamedTuple):
    """Power Iteration과 np.linalg.eig 비교 결과."""

    power_eigenvalue: float
    numpy_eigenvalue: float
    rel_error: float
    cosine: float
    n_iter: int
    passed: bool


def verify_power_iteration(
    matrix: np.ndarray,
    tol_ratio: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> EigenCheck:
    """Power Iteration 결과를 np.linalg.eig 와 비교한다 (eig 는 이 검증에서만 쓴다).

    고유값 상대 오차 |λ_power − λ_eig| / |λ_eig| 가 tol_ratio 이내이고 수렴했으면 통과다.
    고유벡터는 |cos| = |v_power · v_eig| 로 방향이 같은지 본다 (1 에 가까울수록 같다).
    """
    result = power_iteration(matrix, max_iter=max_iter, tol=tol)

    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    idx = int(np.argmax(np.abs(values)))
    ref_value = float(np.real(values[idx]))
    ref_vector = np.real(vectors[:, idx])
    ref_vector = ref_vector / np.linalg.norm(ref_vector)

    rel_error = abs(result.eigenvalue - ref_value) / abs(ref_value)
    cosine = abs(float(result.eigenvector @ ref_vector))
    passed = bool(result.converged and rel_error <= tol_ratio)
    return EigenCheck(
        result.eigenvalue, ref_value, float(rel_error), cosine, result.n_iter, passed
    )


def power_iteration_examples() -> dict[str, np.ndarray]:
    """검증에 쓰는 예제 행렬: 미션 예시(대칭), 비대칭 삼각, 3x3 대칭."""
    return {
        "A = [[4, 1], [1, 3]] (mission, symmetric)": np.array([[4.0, 1.0], [1.0, 3.0]]),
        "B = [[2, 1], [0, 3]] (non-symmetric)": np.array([[2.0, 1.0], [0.0, 3.0]]),
        "C = tridiag(-1, 2, -1) (3x3 symmetric)": np.array(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        ),
    }


def plot_power_iteration(
    examples: dict[str, np.ndarray] | None = None,
    tol: float = 1e-6,
    save_path: Path | str | None = None,
) -> Figure:
    """예제 행렬마다 반복에 따른 방향 변화량이 줄어드는 모습을 로그 축으로 그린다.

    변화량이 직선으로 내려가는 것은 매 반복 오차가 일정한 비율(≈|λ₂/λ₁|)로 줄어든다는 뜻이다.
    """
    examples = power_iteration_examples() if examples is None else examples
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, matrix in examples.items():
        result = power_iteration(matrix, tol=tol)
        ax.semilogy(
            np.arange(1, result.n_iter + 1),
            result.deltas,
            marker="o",
            ms=3,
            label=f"{name}: {result.n_iter} iters",
        )
    ax.axhline(tol, color="0.4", ls="--", lw=1, label=f"tol = {tol:g}")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|v_k - v_{k-1}\|$")
    ax.set_title("Power Iteration convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. SVD 이미지 압축
# ---------------------------------------------------------------------------
def make_sample_image(size: int = 64) -> np.ndarray:
    """NumPy 로 그린 size×size 그레이스케일 풍경 이미지(0~1)를 만든다.

    하늘 그라데이션, 태양, 언덕, 지붕·창문이 있는 집으로 이루어져 있다.
    파일 없이 같은 이미지가 항상 재현되도록 코드로 그리고, 질감 노이즈는 시드 42로 고정한다.
    """
    rng = np.random.default_rng(SEED)
    t = np.linspace(0.0, 1.0, size)
    x, y = np.meshgrid(t, t)  # x: 열 방향(0~1), y: 행 방향(위가 0)

    img = 0.9 - 0.45 * y  # 하늘: 위로 갈수록 밝다
    img = np.where((x - 0.8) ** 2 + (y - 0.2) ** 2 < 0.09**2, 1.0, img)  # 태양
    hill = 0.72 + 0.05 * np.sin(2 * np.pi * 2 * x)
    img = np.where(y > hill, 0.30 + 0.1 * np.sin(2 * np.pi * 6 * x), img)  # 언덕
    img = np.where(
        (x > 0.18) & (x < 0.52) & (y > 0.52) & (y < 0.86), 0.68, img
    )  # 집 몸통
    roof = (y > 0.34) & (y <= 0.52) & (np.abs(x - 0.35) < (y - 0.34) * 1.0)
    img = np.where(roof, 0.18, img)  # 지붕(삼각형)
    img = np.where((x > 0.30) & (x < 0.38) & (y > 0.66) & (y < 0.86), 0.12, img)  # 문
    img = np.where((x > 0.41) & (x < 0.48) & (y > 0.58) & (y < 0.66), 0.95, img)  # 창문

    img = img + 0.02 * rng.normal(size=img.shape)
    return np.clip(img, 0.0, 1.0)


def svd_decompose(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A = U Σ Vᵀ 로 분해한다 (축소형: U (m, r), Σ (r,), Vᵀ (r, n), r = min(m, n))."""
    return np.linalg.svd(np.asarray(image, dtype=float), full_matrices=False)


def svd_reconstruct(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int) -> np.ndarray:
    """특이값 상위 k개만 남긴 rank-k 근사 Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ 를 만든다.

    k 가 rank r = min(m, n) 보다 크면 r 로 줄인다 (그 이상은 더할 특이값이 없다).
    """
    if k < 1:
        raise ValueError(f"k 는 1 이상이어야 한다: {k}")
    k = min(k, s.size)
    return (u[:, :k] * s[:k]) @ vt[:k]


def svd_compress(image: np.ndarray, k: int) -> np.ndarray:
    """이미지를 SVD 로 분해하고 rank-k 로 복원해 돌려준다."""
    u, s, vt = svd_decompose(image)
    return svd_reconstruct(u, s, vt, k)


class SvdStats(NamedTuple):
    """rank-k 근사의 품질과 저장량.

    Attributes:
        k: 요청한 k
        k_eff: 실제로 쓴 k (= min(k, rank))
        rel_error: 상대 프로베니우스 오차 ‖A − Aₖ‖_F / ‖A‖_F
        energy: 남긴 에너지 비율 Σσᵢ² (i ≤ k) / Σσᵢ²
        storage: 저장량 비율 k(m + n + 1) / (m·n)  (U의 k열, V의 k열, σ k개)
    """

    k: int
    k_eff: int
    rel_error: float
    energy: float
    storage: float


def svd_stats(image: np.ndarray, k: int) -> SvdStats:
    """rank-k 근사의 오차, 에너지, 저장량을 계산한다.

    Eckart–Young 정리에 따라 ‖A − Aₖ‖_F² = Σ_{i>k} σᵢ² 이므로 오차는 버린 특이값의 에너지와 같다.
    """
    image = np.asarray(image, dtype=float)
    u, s, vt = svd_decompose(image)
    approx = svd_reconstruct(u, s, vt, k)
    k_eff = min(k, s.size)
    m, n = image.shape
    rel_error = float(np.linalg.norm(image - approx) / np.linalg.norm(image))
    energy = float(np.sum(s[:k_eff] ** 2) / np.sum(s**2))
    storage = k_eff * (m + n + 1) / (m * n)
    return SvdStats(k, k_eff, rel_error, energy, float(storage))


def plot_svd_comparison(
    image: np.ndarray | None = None,
    ks: tuple[int, ...] = (10, 50, 100),
    save_path: Path | str | None = None,
) -> Figure:
    """원본과 k = 10, 50, 100 복원 결과를 한 Figure에 나란히 비교한다."""
    image = make_sample_image() if image is None else image
    u, s, vt = svd_decompose(image)

    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3.4 * (len(ks) + 1), 3.9))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original {image.shape[0]}x{image.shape[1]}\nrank ≤ {s.size}")
    for ax, k in zip(axes[1:], ks, strict=True):
        stats = svd_stats(image, k)
        approx = np.clip(svd_reconstruct(u, s, vt, k), 0, 1)
        ax.imshow(approx, cmap="gray", vmin=0, vmax=1)
        label = (
            f"k = {k}" if stats.k_eff == k else f"k = {k} → {stats.k_eff} (full rank)"
        )
        ax.set_title(
            f"{label}\nerror {stats.rel_error:.1%} · storage {stats.storage:.0%}"
        )
    for ax in axes:
        ax.axis("off")
    fig.suptitle("SVD low-rank reconstruction")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, save_path)
    return fig


def plot_singular_values(
    image: np.ndarray | None = None, save_path: Path | str | None = None
) -> Figure:
    """특이값 σᵢ 와 누적 에너지를 그려, 왜 적은 k 로도 이미지가 복원되는지 보여 준다."""
    image = make_sample_image() if image is None else image
    _, s, _ = svd_decompose(image)
    energy = np.cumsum(s**2) / np.sum(s**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogy(np.arange(1, s.size + 1), s, marker="o", ms=3)
    ax1.set_xlabel("index i")
    ax1.set_ylabel(r"singular value $\sigma_i$")
    ax1.set_title("Singular values")
    ax1.grid(alpha=0.3)

    ax2.plot(np.arange(1, s.size + 1), energy, marker="o", ms=3)
    ax2.axhline(0.99, color="0.5", ls="--", lw=1)
    ax2.set_xlabel("k")
    ax2.set_ylabel(r"$\sum_{i\leq k}\sigma_i^2 / \sum_i \sigma_i^2$")
    ax2.set_title("Cumulative energy")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def _save(fig: Figure, save_path: Path | str | None) -> None:
    """save_path 가 있으면 폴더를 만들고 PNG 로 저장한다."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    """모든 그래프를 outputs/ 에 저장하고 검증 결과를 표로 출력한다."""
    np.random.seed(SEED)

    print("== 행렬식 vs 면적비 (허용 오차 1%) ==")
    extra = {
        "S(2, 1.5): det=3": scaling_matrix(2, 1.5),
        "singular [[1,2],[2,4]]: det=0": np.array([[1.0, 2.0], [2.0, 4.0]]),
    }
    for name, matrix in {**default_transforms(), **extra}.items():
        c = check_area_ratio(matrix)
        print(
            f"{name:32s} det={c.det:7.3f}  area ratio={c.area_ratio:7.3f}  error={c.rel_error:.2e}  {'OK' if c.passed else 'FAIL'}"
        )
    plot_transformations(save_path=OUTPUT_DIR / "linear_transformations.png")

    print("\n== Power Iteration vs np.linalg.eig (허용 오차 5%) ==")
    for name, matrix in power_iteration_examples().items():
        c = verify_power_iteration(matrix)
        print(
            f"{name:44s} λ_power={c.power_eigenvalue:.4f}  λ_eig={c.numpy_eigenvalue:.4f}  "
            f"error={c.rel_error:.2e}  |cos|={c.cosine:.6f}  iters={c.n_iter}  {'OK' if c.passed else 'FAIL'}"
        )
    plot_power_iteration(save_path=OUTPUT_DIR / "power_iteration_convergence.png")

    print("\n== SVD 압축 (64x64) ==")
    image = make_sample_image()
    for k in (10, 50, 100):
        st = svd_stats(image, k)
        print(
            f"k={k:3d} (effective {st.k_eff:2d})  error={st.rel_error:6.2%}  energy={st.energy:7.3%}  storage={st.storage:6.1%}"
        )
    plot_svd_comparison(image, save_path=OUTPUT_DIR / "svd_compression.png")
    plot_singular_values(image, save_path=OUTPUT_DIR / "svd_singular_values.png")
    plt.close("all")


if __name__ == "__main__":
    main()
