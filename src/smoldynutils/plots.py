from typing import Optional, Sequence, Union, Dict

import numpy as np
from matplotlib.axes import Axes

from smoldynutils.data_objects import Trajectory, TrajectorySet

import seaborn as sns

FloatArray = np.typing.NDArray[np.floating]


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

    for traj in trajs:
        ax = plot_trajectorie(traj, ax, title)
    return ax


def plot_msd(
    msd: np.ndarray,
    ax: Axes,
    time: Optional[np.ndarray] = None,
    title: str = "Title",
    color: Optional[str] = None,
) -> Axes:
    if time is None:
        time = np.arange(len(msd))
    if color is None:
        color = "blue"
    if len(msd.shape) > 2:
        raise ValueError("Input MSD array is > 2D")
    if not len(msd) == len(time):
        msd = msd.T
        if not len(msd) == len(time):
            raise ValueError("Input MSD array and time array have no shape in common.")
    ax.plot(time, msd, color=color)
    ax.set_xlabel("time")
    ax.set_ylabel("msd")
    ax.set_title(title)
    return ax


def plot_msd_comparison(
    msd: np.ndarray,
    theoretical_msd: np.ndarray,
    ax: Axes,
    time: Optional[np.ndarray] = None,
    title: str = "Title",
) -> Axes:
    """Lineplot showing the calculated MSD values vs the theoretical expectation.

    Args:
        msd (np.ndarray): Calculated MSD values
        theoretical_msd (np.ndarray): Theoretically expected MSD values
        ax (Axes): Axis onto which will be plotted
        title (str, optional): Figure title. Defaults to "Title".

    Returns:
        Axes: Axis that contains the msd comparison.
    """
    if len(msd) != len(theoretical_msd):
        raise ValueError(
            f"Mismatch in MSD arrays: input array length={len(msd)}, theoretical array length={len(theoretical_msd)}"
        )
    if time is None:
        time = np.arange(len(msd))
    ax = plot_msd(msd, ax, time=time)
    ax = plot_msd(theoretical_msd, ax, time=time, color="red")
    ax.set_xlabel("time")
    ax.set_ylabel("msd")
    ax.set_title(title)
    return ax


def plot_diffconst_hist(
    diffcoffs: np.ndarray, reference_diffcoff: float, ax: Axes, title: str = "Title"
) -> Axes:
    """Plots histogram of diffusion coefficients

    Args:
        diffcoffs (np.ndarray): Array of diffusion coefficients
        reference_diffcoff (float): Expected diffusion coefficient
        ax (Axes): Axes onto which will be plotted
        title (str, optional): Plot title. Defaults to "Title".

    Returns:
        Axes: Axes with histogram
    """

    lower_bound = min(diffcoffs)
    upper_bound = max(diffcoffs)
    bins = list(np.linspace(lower_bound, upper_bound, 20))
    ax.hist(diffcoffs, bins=bins)
    ax.set_xscale("log")

    ax.axvline(float(np.mean(diffcoffs)))

    ax.axvline(reference_diffcoff)

    ax.set_xlabel("Diffusion coefficient")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def plot_violin_with_mean(
    diffcoff: Dict[float, FloatArray],
    reference_diffcoffs: Sequence[float],
    permeability: Sequence[float],
    ax: Axes,
    title: str = "Title",
) -> Axes:
    """Generates a violinplot of diffcoff vs permeability.

    Args:
        diffcoff (Dict[float, FloatArray]): Permeability vs diffusion coefficients
        reference_diffcoffs (Sequence[float]): Expected diffusion coefficients
        permeability (Sequence[float]): Permeabilities for x axis
        ax (Axes): Axes onto which will be plotted
        title (str, optional): Title of plot. Defaults to "Title".

    Raises:
        ValueError: Number of entries in diffcoff does not match number of permeabilities

    Returns:
        Axes: Axis that contains violin plots
    """
    if not len(diffcoff.keys()) == len(permeability):
        raise ValueError(
            "Number of entries in diffcoff dict does not match number of permeabilites."
        )
    sns.violinplot(diffcoff, ax=ax, order=list(diffcoff.keys()), color="skyblue", inner=None)
    mean_ds = [np.mean(vals) for vals in diffcoff.values()]
    indices = np.arange(0, len(diffcoff.keys()))
    ax.scatter(indices, mean_ds, color="black", marker="_", zorder=10, alpha=1, s=100)
    ax.axhline(
        reference_diffcoffs[0],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"WT D={reference_diffcoffs[0]}",
    )
    ax.axhline(
        reference_diffcoffs[1],
        color="blue",
        linestyle=":",
        linewidth=1,
        label=f"PHSD D={reference_diffcoffs[1]}",
    )
    ax.set_xlabel("Permeability")
    ax.set_ylabel("Diffusion coefficient")
    ax.set_title(title)
    return ax
