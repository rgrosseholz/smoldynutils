import numpy as np


def gauss_probability_density(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    value = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-np.square(x - mu) / (2 * sigma**2))
    return float(value)


def theoretical_brownian_motion_pdf(x: float, D: float, t: float) -> float:
    if D <= 0:
        raise ValueError("D must be > 0")
    if t < 0:
        raise ValueError("t must be >= 0")
    sigma = np.sqrt(2 * D * t)
    mu = 0
    return gauss_probability_density(x, mu, sigma)


def theoretical_msd(t: float, D: float) -> float:
    return 4 * D * t


def theoretical_msd_residue(t: float, D: float, epsilon: float) -> float:
    return 4 * D * t + epsilon
