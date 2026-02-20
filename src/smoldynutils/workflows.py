from typing import Dict, Sequence

import numpy as np

from smoldynutils.data_objects import Trajectory, TrajectorySet
from smoldynutils.metrics import (
    calc_combined_msd,
    calc_sq_displacement_from_zero,
    calc_xy_displacement,
    calc_xy_msd,
    estimate_diffcoff,
)


def estimate_timelag_msd_from_traj(traj: Trajectory, timelags: Sequence[int]) -> Dict[int, float]:
    """Calculates MSD(timelag) for trajectory.

    Args:
        traj (Trajectory): Trajectory for which MSD will be calculated
        timelags (Sequence[int]): Sequence of timelags that will be used

    Returns:
        Dict[int, float]: Keys are timelags, values the corresponding MSD
    """
    msd_dict = {}
    for timelag in timelags:
        xy_displacement = calc_xy_displacement(traj, timelag)
        xy_msd = calc_xy_msd(xy_displacement)
        msd = calc_combined_msd(xy_msd)
        msd_dict[timelag] = msd
    return msd_dict


def estimate_timelag_diffcoff_from_trajset(
    trajs: TrajectorySet, timelags: Sequence[int] = (1, 2, 3, 4)
) -> Dict[int, float]:
    """Calculates observed diffusion coefficient based on MSD(timelag) for set of trajectories

    Args:
        trajs (TrajectorySet): Set of trajectories for which diff coff will be calculated
        timelags (Sequence[int], optional): Sequence of timelags for MSD calculation. Defaults to (1, 2, 3, 4).

    Returns:
        Dict[int, float]: Keys are trajectory serialnums or index, values the corresponding diff coff.
    """
    diffcoffs = {}
    use_index_for_dict = False
    timelag_array = np.array(timelags)
    if len(np.unique(trajs.serialnums)) < len(trajs):
        use_index_for_dict = True
    for index, traj in enumerate(trajs):
        msd_dict = estimate_timelag_msd_from_traj(traj, timelags)
        msds = np.array(list(msd_dict.values()))
        if use_index_for_dict is True:
            diffcoffs[index] = estimate_diffcoff(msds, timelag_array)
        else:
            diffcoffs[traj.serialnumber] = estimate_diffcoff(msds, timelag_array)
    return diffcoffs


def estimate_time_msd_from_traj(traj: Trajectory) -> np.ndarray:
    """Calculates MSD(time) for trajectory.

    Args:
        traj (Trajectory): Trajectory for which MSD will be calculated

    Returns:
        np.ndarray: Calculated MSDs.
    """
    x_sqdisplacement = calc_sq_displacement_from_zero(traj.x)
    y_sqdisplacement = calc_sq_displacement_from_zero(traj.y)
    msd = calc_combined_msd((x_sqdisplacement, y_sqdisplacement))
    return msd


def estimate_time_diffcoff_from_trajset(trajs: TrajectorySet) -> Dict[int, float]:
    """Estimates diffusion coefficient of set of Trajectories based on MSD(time)

    Args:
        trajs (TrajectorySet): Set of trajectories for which diffusion coefficient will be estimated.

    Returns:
        Dict[int, float]: Keys are serialnums or index, values are diffcoffs
    """
    diffcoffs = {}
    use_index_for_dict = False
    if len(np.unique(trajs.serialnums)) < len(trajs):
        use_index_for_dict = True
    for index, traj in enumerate(trajs):
        msd = estimate_time_msd_from_traj(traj)
        if use_index_for_dict is True:
            diffcoffs[index] = estimate_diffcoff(msd, traj.t)
        else:
            diffcoffs[traj.serialnumber] = estimate_diffcoff(msd, traj.t)
    return diffcoffs
