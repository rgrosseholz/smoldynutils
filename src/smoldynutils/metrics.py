import numpy as np

from smoldynutils.data_objects import Trajectory


def calc_displacements(traj_values: np.ndarray, lag: int = 1) -> np.ndarray:
    """Calculates the displacement depending on time lag.

    Eq: x(t) - x(t+lag)

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
    if lag > len(traj.x) or lag > len(traj.y):
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


def calc_sq_displacement_from_zero(traj_values: np.ndarray) -> np.ndarray:
    """Calculates displacement relative to start position.

    Args:
        traj_values (np.ndarray): Position value of Trajectory

    Returns:
        np.ndarray: Displacement from start position.
    """
    return (traj_values - traj_values[0]) ** 2


def calc_combined_msd(msds: tuple[np.ndarray, np.ndarray]) -> float:
    return NotImplemented
