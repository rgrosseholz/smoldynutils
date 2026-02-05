import numpy as np
import pytest
from smoldynutils.utils import *


def test_gaussian_probability():
    mu = 0
    sigma = 1
    expected = 1 / np.sqrt(2 * np.pi)
    assert np.isclose(gauss_probability_density(0, mu, sigma), expected)

    assert np.isclose(
        gauss_probability_density(-1, mu, sigma), gauss_probability_density(1, mu, sigma)
    )


@pytest.mark.parametrize("sigma", [0, -1])
def test_gaussian_raises_error(sigma):
    with pytest.raises(ValueError):
        gauss_probability_density(0, 0, sigma=sigma)


def test_theoretical_brownian_motion():
    D = 2
    t = 3
    x = 0.5

    sigma = np.sqrt(2 * D * t)

    assert np.isclose(
        theoretical_brownian_motion_pdf(x, D, t), gauss_probability_density(x, 0, sigma)
    )


@pytest.mark.parametrize("D, t", [(-1, 1), (1, -1), (1, 0)])
def test_brownian_error(D, t):
    with pytest.raises(ValueError):
        theoretical_brownian_motion_pdf(0, D, t)


def test_theoretical_msds():
    assert theoretical_MSD(0, 0) == 0
    assert theoretical_MSD(1, 1) == 4

    assert theoretical_MSD_residue(0, 0, 0) == 0
    assert theoretical_MSD_residue(1, 1, 1) == 5
