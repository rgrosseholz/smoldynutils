import numpy as np


def gauss_probability_density(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-np.square(x - mu) / (2 * sigma**2))


def theoretical_brownian_motion_pdf(x, D, t):
    if D <= 0:
        raise ValueError("D must be > 0")
    if t < 0:
        raise ValueError("t must be >= 0")
    sigma = np.sqrt(2 * D * t)
    mu = 0
    return gauss_probability_density(x, mu, sigma)


def theoretical_MSD(t, D):
    return 4 * D * t


def theoretical_MSD_residue(t, D, epsilon):
    return 4 * D * t + epsilon
