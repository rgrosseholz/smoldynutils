import warnings
from typing import cast

import numpy as np
from scipy.optimize import curve_fit

from smoldynutils.data_objects import Trajectory
from smoldynutils.utils import theoretical_msd, theoretical_msd_residue

FloatArray = np.typing.NDArray[np.floating]


def calc_displacements(traj_values: FloatArray, lag: int = 1) -> FloatArray:
    """Calculates the displacement depending on time lag.

    Eq: x(t+lag) - x(t)

    Args:
        traj_values (np.ndarray): x or y values
        lag (int, optional): Controls the shift of the window. Defaults to 1.

    Raises:
        ValueError: Chosen timelag is bigger than the length of x/y

    Returns:
        np.ndarray: Timelag displacement values
    """
    if lag > len(traj_values) - 1:
        raise ValueError("Timelag is bigger than length of trajectory.")
    displacement = traj_values[lag:] - traj_values[:-lag]
    return displacement


def calc_xy_displacement(traj: Trajectory, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Feeds x and y of Trajectory into calc_displacements

    Args:
        traj (Trajectory): Trajectory object for which x and y displacements should be calculated.
        lag (int, optional): Controls the shift of the window. Defaults to 1.

    Raises:
        ValueError: Chosen Timelag is bigger than the trajectory is long.

    Returns:
        tuple[np.ndarray, np.ndarray]: Timelag displacements in x and y direction
    """
    if lag > len(traj.x) - 1 or lag > len(traj.y) - 1:
        raise ValueError("Timelag is bigger than number of datapoints in x or y")
    x_displacement = calc_displacements(traj.x, lag)
    y_displacement = calc_displacements(traj.y, lag)

    return (x_displacement, y_displacement)


def calc_msd(displacment: np.ndarray) -> np.ndarray:
    """Calculates mean squeared displacement.

    Equation: mean(dx**2)

    Args:
        displacment (np.ndarray): Displacement values.

    Returns:
        np.ndarray: Mean squared displacement values.
    """
    squared_displacement = displacment**2
    mean_squared_displacement = np.array(np.mean(squared_displacement))
    return mean_squared_displacement


def calc_xy_msd(displacements: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Feeds x and y into calc_msd

    Args:
        displacements (tuple[np.ndarray, np.ndarray]): x displacement followed by y displacement

    Returns:
        tuple[np.ndarray, np.ndarray]: MSD of x and y
    """
    x_msd = calc_msd(displacements[0])
    y_msd = calc_msd(displacements[1])
    return (x_msd, y_msd)


def calc_sq_displacement_from_zero(traj_values: FloatArray) -> FloatArray:
    """Calculates displacement relative to start position.

    Args:
        traj_values (np.ndarray): Position value of Trajectory

    Returns:
        np.ndarray: Displacement from start position.
    """
    x0 = float(traj_values[0])
    return (traj_values - x0) ** 2


def calc_combined_msd(msds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.array(msds[0] + msds[1])


def estimate_diffcoff_fullinfo(
    msds: np.ndarray, timepoints: np.ndarray, add_epsilon: bool = False
) -> tuple[FloatArray, FloatArray]:
    """Estimates diffusion coefficient from MSD.

    Fitted equation is MSD = 4*D*t

    Args:
        msds (np.ndarray): Array of MSD values
        timepoints (np.ndarray): Array of timelag or time values
        add_epsilon (bool, optional): Use equation MSD = 4*D*t + epsilon for fitting. Defaults to False.
        return_full (bool, optional): Return full information about curve fitting. Defaults to False.

    Returns:
        np.ndarray: _description_
    """

    if len(timepoints) < 2 and add_epsilon is True:
        warnings.warn(
            "Cannot fit with epsilon if only one timelag given. Setting add_epsilon to False.",
            UserWarning,
        )
        add_epsilon = False
    if add_epsilon is True:
        line_fit = curve_fit(theoretical_msd_residue, timepoints, msds)
    else:
        line_fit = curve_fit(theoretical_msd, timepoints, msds)
    return cast(tuple[FloatArray, FloatArray], line_fit)


def estimate_diffcoff(msds: np.ndarray, timepoints: np.ndarray, add_epsilon: bool = False) -> float:
    """Estimates diffusion coefficient from MSD.

    Fitted equation is MSD = 4*D*t

    Args:
        msds (np.ndarray): Array of MSD values
        timepoints (np.ndarray): Array of timelag or time values
        add_epsilon (bool, optional): Use equation MSD = 4*D*t + epsilon for fitting. Defaults to False.
        return_full (bool, optional): Return full information about curve fitting. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    popt, _ = estimate_diffcoff_fullinfo(msds, timepoints, add_epsilon)
    return float(popt[0])
