from typing import Sequence, Union

import numpy as np
from matplotlib.axes import Axes

from smoldynutils.data_objects import Trajectory, TrajectorySet


def plot_gauss_comparison(
    displacement: np.ndarray,
    gauss_vals: np.ndarray,
    ax: Axes,
    bins: Union[str, Sequence[float]] = "fd",
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


def plot_trajectorie(traj: Trajectory, ax: Axes, title: str = "Title") -> Axes:
    """Simple xy plot of a single trajectory.

    Args:
        traj (Trajectory): Trajectory to plot
        ax (Axes): Axis onto which the trajectory will be plotted
        title (str, optional): Figure title. Defaults to "Title".

    Raises:
        ValueError: No axis to plot onto provided

    Returns:
        Axes: Axis that contains the xy plot
    """
    if ax is None:
        raise ValueError("No axis to plot onto provided.")
    ax.plot(traj.x, traj.y)
    return ax


def plot_trajectories(trajs: TrajectorySet, ax: Axes, title: str = "Title") -> Axes:
    """Creates xy plot for multiple trajectories.

    Args:
        trajs (TrajectorySet): Set of trajectories
        ax (Axes): Axis onto which the trajectory will be plotted
        title (str, optional): Figure title. Defaults to "Title".

    Raises:
        ValueError: No axis to plot onto provided

    Returns:
        Axes: Axis that contains the xy plot
    """
    if ax is None:
        raise ValueError("No axis to plot onto provided.")
    for traj in trajs:
        plot_trajectorie(traj, ax, title)
    return ax
