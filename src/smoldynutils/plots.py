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
    ax.plot(traj.x, traj.y, color="black")
    ax.scatter(traj.x, traj.y, c=traj.t)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
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


def plot_msd(msd: Sequence[float], ax: Axes, title: str = "Title") -> Axes:
    ax.plot(msd, color="blue")
    ax.set_xlabel("time")
    ax.set_ylabel("msd")
    ax.set_title(title)
    return ax


def plot_msd_comparison(
    msd: np.ndarray, theoretical_msd: np.ndarray, ax: Axes, title: str = "Title"
) -> Axes:
    """Lineplot showing the calculated MSD values vs the theoretical expectation.

    Args:
        msd (np.ndarray): Calculated MSD values
        theoretical_msd (np.ndarray): Theoretically expected MSD values
        ax (Axes): Axis onto which will be plotted
        title (str, optional): Figure title. Defaults to "Title".

    Raises:
        ValueError: No axis to plot onto provided.

    Returns:
        Axes: Axis that contains the msd comparison.
    """
    if ax is None:
        raise ValueError("No axis to plot onto provided.")
    ax = plot_msd(msd, ax)
    ax.plot(theoretical_msd, color="red")
    ax.set_xlabel("time")
    ax.set_ylabel("msd")
    ax.set_title(title)
    return ax


def plot_diffconst_estimation(
    diffcoffs: np.ndarray, reference_diffcoff: float, ax: Axes, title: str = "Title"
) -> Axes:
    if ax is None:
        raise ValueError("No axis to plot onto provided.")
    lower_bound = min(diffcoffs)
    upper_bound = max(diffcoffs)

    ax.hist(diffcoffs, bins=np.geomspace(lower_bound, upper_bound, 20))
    ax.set_xscale("log")

    ax.axvline(float(np.mean(diffcoffs)))

    ax.axvline(reference_diffcoff)

    ax.set_xlabel("Diffusion coefficient")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax
