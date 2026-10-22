"""확률분포, Softmax, 손실 함수와 정보 이론 도구 (A2-1).

- 정규분포 PDF, 베르누이 PMF, Softmax
- MSE, 이진/범주형 Cross-Entropy와 음의 로그 가능도(NLL)
- 보너스: 엔트로피, Cross-Entropy, KL-Divergence 직접 구현

수식 유도와 MLE 연결은 `notebooks/probability_loss.ipynb`에 있고, 이 모듈은 그 노트북이
호출하는 계산을 담는다. 로그의 밑은 모두 자연로그(단위 nat)다.

실행: 프로젝트 폴더(`projects/a2-1-ai-math`)에서 ``uv run python -m src.probability``
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# 1. 분포
# ----------------------------------------------------------------------
def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    r"""정규분포 N(μ, σ²)의 확률밀도함수.

    .. math:: f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    PDF의 값은 확률이 아니라 밀도라서 1보다 클 수 있다. 구간에서 적분해야 확률이 된다.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))


def bernoulli_pmf(k: np.ndarray, p: float) -> np.ndarray:
    r"""베르누이 분포 Bernoulli(p)의 확률질량함수 :math:`P(K=k) = p^k (1-p)^{1-k},\ k \in \{0, 1\}`."""
    k = np.asarray(k)
    return np.where(k == 1, p, 1.0 - p).astype(float)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""Softmax :math:`p_i = e^{z_i} / \sum_j e^{z_j}`.

    ``z``에서 최댓값을 먼저 빼도 결과가 같다(분자·분모에 같은 ``e^{-max}``가 곱해진다).
    이렇게 하면 ``exp``의 입력이 모두 0 이하가 되어 ``z = 1000``처럼 큰 값에서도 넘치지 않는다.
    """
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""안정적인 :math:`\log p_i = z_i - \max z - \log\sum_j e^{z_j - \max z}`."""
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


# ----------------------------------------------------------------------
# 2. 손실 함수와 음의 로그 가능도
# ----------------------------------------------------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""평균 제곱 오차 :math:`\frac1n \sum_i (y_i - \hat y_i)^2`."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff**2))


def binary_cross_entropy(y_true: np.ndarray, p: np.ndarray) -> float:
    r"""이진 교차 엔트로피 :math:`-\frac1n \sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]`."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def categorical_cross_entropy(y_onehot: np.ndarray, probs: np.ndarray) -> float:
    r"""범주형 교차 엔트로피 :math:`-\frac1n \sum_i \sum_c y_{ic} \log p_{ic}` (행이 샘플)."""
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return float(-np.mean(np.sum(np.asarray(y_onehot) * np.log(probs), axis=-1)))


def softmax_cross_entropy_grad(logits: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    r"""Softmax + Cross-Entropy의 logits에 대한 기울기 :math:`\partial L / \partial z = p - y`.

    Softmax의 야코비안 ``∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)``와 ``∂L/∂pᵢ = -yᵢ/pᵢ``를 곱해 정리하면
    ``Σᵢ(-yᵢ/pᵢ)·pᵢ(δᵢⱼ - pⱼ) = -yⱼ + pⱼ Σᵢ yᵢ = pⱼ - yⱼ`` (원-핫이면 Σᵢ yᵢ = 1)다.
    """
    return softmax(logits) - np.asarray(y_onehot, dtype=float)


def normal_nll(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    r"""정규분포 가정 아래 데이터 y의 음의 로그 가능도(합).

    .. math:: -\sum_i \log N(y_i;\mu_i,\sigma^2)
        = \frac n2\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i (y_i-\mu_i)^2

    σ가 고정이면 둘째 항(= n·MSE/2σ²)만 μ에 의존한다. 즉 NLL 최소화 = MSE 최소화다.
    """
    y = np.asarray(y, dtype=float)
    residual = y - np.asarray(mu, dtype=float)
    return float(
        0.5 * y.size * np.log(2.0 * np.pi * sigma**2)
        + np.sum(residual**2) / (2.0 * sigma**2)
    )


def bernoulli_nll(y: np.ndarray, p: np.ndarray) -> float:
    r"""베르누이 가정 아래 데이터 y의 음의 로그 가능도(합) = ``n × BCE``."""
    y = np.asarray(y, dtype=float)
    return float(y.size * binary_cross_entropy(y, p))


def categorical_nll(y_index: np.ndarray, probs: np.ndarray) -> float:
    """범주형 가정 아래 정답 클래스 번호 ``y_index``의 음의 로그 가능도(합)."""
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    rows = np.arange(len(y_index))
    return float(-np.sum(np.log(probs[rows, np.asarray(y_index)])))


def normal_mle(y: np.ndarray) -> tuple[float, float]:
    r"""정규분포 MLE: :math:`\hat\mu = \bar y,\ \hat\sigma^2 = \frac1n\sum(y_i-\bar y)^2`.

    분산 MLE는 n으로 나눈다(편향 추정량). 표본분산(n-1)과 다르다.
    """
    y = np.asarray(y, dtype=float)
    mu = float(np.mean(y))
    return mu, float(np.sqrt(np.mean((y - mu) ** 2)))


def bernoulli_mle(y: np.ndarray) -> float:
    r"""베르누이 MLE: :math:`\hat p = \frac1n\sum y_i`, 즉 1의 비율."""
    return float(np.mean(np.asarray(y, dtype=float)))


# ----------------------------------------------------------------------
# 3. 정보 이론 (보너스)
# ----------------------------------------------------------------------
def _check_distribution(p: np.ndarray, name: str) -> np.ndarray:
    """음수가 없고 합이 1인지 확인하고 float 배열로 돌려준다. 아니면 ValueError."""
    arr = np.asarray(p, dtype=float)
    if np.any(arr < 0) or not np.isclose(arr.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"{name}는 음수가 없고 합이 1인 확률분포여야 합니다 (합={arr.sum():.6f})."
        )
    return arr


def entropy(p: np.ndarray) -> float:
    r"""엔트로피 :math:`H(p) = -\sum_i p_i \log p_i` (nat). ``0 log 0 = 0``으로 둔다.

    분포가 균등할수록 크고(최대 ``log n``), 한 점에 몰릴수록 0에 가깝다.
    """
    p = _check_distribution(p, "p")
    nonzero = p > 0
    return (
        float(-np.sum(p[nonzero] * np.log(p[nonzero]))) + 0.0
    )  # 한 점에 몰린 분포의 -0.0을 0.0으로


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    r"""교차 엔트로피 :math:`H(p, q) = -\sum_i p_i \log q_i`.

    진짜 분포 p의 데이터를 q로 부호화할 때 필요한 평균 정보량이다. ``p_i > 0``인데
    ``q_i = 0``이면 무한대이므로 ``q``를 ``EPS``로 잘라 계산한다.
    """
    p = _check_distribution(p, "p")
    q = np.clip(_check_distribution(q, "q"), EPS, 1.0)
    nonzero = p > 0
    return float(-np.sum(p[nonzero] * np.log(q[nonzero])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL 발산 :math:`D_{KL}(p\,\|\,q) = \sum_i p_i \log\frac{p_i}{q_i} = H(p,q) - H(p)`.

    항상 0 이상이고 p = q일 때만 0이다(Gibbs 부등식). 대칭이 아니어서
    ``D(p‖q) ≠ D(q‖p)``이므로 거리가 아니다. 분류에서 p는 정답(원-핫)이라 ``H(p) = 0``이고,
    따라서 Cross-Entropy 최소화 = KL 최소화다.
    """
    p = _check_distribution(p, "p")
    q = np.clip(_check_distribution(q, "q"), EPS, 1.0)
    nonzero = p > 0
    return float(np.sum(p[nonzero] * np.log(p[nonzero] / q[nonzero])))


def gaussian_kl(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    r"""두 정규분포의 KL 발산 해석해 :math:`D_{KL}(N_1\|N_2)`.

    .. math:: \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12
    """
    return float(
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2.0 * sigma2**2)
        - 0.5
    )


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def main() -> None:
    """분포의 정규화, Softmax 합, 정보 이론 항등식을 빠르게 확인해 출력한다."""
    np.random.seed(42)

    grid = np.linspace(-12.0, 12.0, 24001)
    dx = grid[1] - grid[0]
    for mu, sigma in ((0.0, 1.0), (2.0, 0.5)):
        area = float(np.sum(normal_pdf(grid, mu, sigma)) * dx)
        print(f"N({mu}, {sigma}²) PDF 적분 = {area:.6f}")
    for p in (0.5, 0.8):
        print(f"Bernoulli({p}) PMF 합 = {bernoulli_pmf(np.array([0, 1]), p).sum():.6f}")

    logits = np.random.randn(1000, 10) * 5.0
    deviation = float(np.max(np.abs(softmax(logits).sum(axis=1) - 1.0)))
    print(
        f"Softmax 출력 합의 1.0과의 최대 오차 = {deviation:.2e} (허용 1e-6)  {'통과' if deviation < 1e-6 else '실패'}"
    )
    big = softmax(np.array([1000.0, 1001.0, 1002.0]))
    print(f"z=[1000, 1001, 1002]에서도 합 = {big.sum():.6f}, 값 = {np.round(big, 4)}")

    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    print(
        f"H(p)={entropy(p):.4f}  H(p,q)={cross_entropy(p, q):.4f}  KL(p||q)={kl_divergence(p, q):.4f}"
    )
    print(
        f"H(p,q) = H(p) + KL(p||q)?  {np.isclose(cross_entropy(p, q), entropy(p) + kl_divergence(p, q))}"
    )


if __name__ == "__main__":
    main()
