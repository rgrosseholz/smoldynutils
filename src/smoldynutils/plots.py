from typing import Union

import numpy as np
from matplotlib.axes import Axes


def plot_gauss_comparison(
    displacement: np.ndarray,
    gauss_vals: np.ndarray,
    ax: Axes,
    bins: Union[str, np.ndarray] = "fd",
    title: str = "Title",
) -> Axes:
    """Plots histogram of measured displacement and theoretical displacement.

    Args:
        displacement (np.ndarray): Measured displacement
        gauss_vals (np.ndarray): Theoretical expectation
        ax (Axes, optional): Axis to plot onto. Defaults to None.
        bins (str, optional): Algorithm to determine bins or bins. Defaults to "fd".
        title (str, optional): Title for the plot. Defaults to "Title".

    Raises:
        ValueError: No axis to plot onto provided.

    Returns:
        Axes: Axis that contains the histogram.
    """
    if ax is None:
        raise ValueError("No axis to plot onto provided.")
    ax.hist(displacement, bins=bins, density=True)
    ax.hist(gauss_vals, bins=bins, density=True)
    ax.set_xlabel("Δx")
    ax.set_ylabel("density")
    ax.set_title(title)
    return ax
