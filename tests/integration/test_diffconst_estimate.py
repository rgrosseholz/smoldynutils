import numpy as np
import pytest

from smoldynutils.metrics import *

expected_d = 0.5


@pytest.fixture
def tau_traj():
    # Finding an artificial trajectory that produces an exact D using tau method
    # is not trivial.
    # Relation between tau and MSD is = 4Dtau; we give a D and know tau
    # For D = 0.5 we get MSD(tau) = 2*tau
    # For tau = [1, 2, 3] the trajectory must satisfy:
    # <(x(n) - x(n-1))^2 + (y(n) - y(n-1))^2> = 2*tau = 2
    # <(x(n) - x(n-2))^2 + (y(n) - y(n-2))^2> = 2*tau = 4
    # <(x(n) - x(n-3))^2 + (y(n) - y(n-3))^2> = 2*tau = 6
    # The minimal trajectory has a length of 4 so max n = 3 and we set x(0) and y(0) to 0
    # With d_ij = (xj-xi)^2 + (yj + yi)^2 we get the following three equations:
    # d_03                   = 6 # tau=3
    # (d_02 + d_13)/2        = 4 # tau=2
    # (d_01 + d_12 + d_23)/3 = 2 # tau=1
    # Assuming all tau steps to be equal
    # this can be solved to yield the below values as one possible solution

    s = np.sqrt(2)
    t = np.array([0, 1, 2, 3], dtype=np.float32)
    x = np.array([0, s, s + 1, s + 1], dtype=np.float32)
    y = np.array([0, 0, -1, s - 1], dtype=np.float32)
    species = np.array([1, 1, 1, 1], dtype=np.uint8)
    return Trajectory(1, x=x, y=y, t=t, species=species)


@pytest.fixture
def time_traj():
    # Finding an artificial trajectory that produces an exact D using t is more
    # straightforward.
    # Equations are largely the same for D=0.5, apart from the mean calculation.
    # We thus get:
    # d_03 = 6
    # d_02 = 4
    # d_01 = 2
    # These equations define three circles in the xy plane along which the conditions
    # are fulfilled.
    # If we let |x3| = |y3| (and same for 2 and 1), then we get
    # x3^2 + y3^2 = 6 <=> 2*|x3|^2 = 6 <=> |x3| =  sqrt(3)
    val_2 = np.sqrt(2)
    val_3 = np.sqrt(3)
    t = np.array([0, 1, 2, 3], dtype=np.float32)
    x = np.array([0, 1, val_2, val_3], dtype=np.float32)
    y = np.array([0, -1, -val_2, -val_3], dtype=np.float32)
    species = np.array([1, 1, 1, 1], dtype=np.uint8)
    return Trajectory(1, x=x, y=y, t=t, species=species)


def test_dummy_trajectory_timelag(tau_traj):
    data_dict = {}
    time_lags = np.array([1, 2, 3])
    for time_lag in time_lags:
        xy_displacement = calc_xy_displacement(tau_traj, time_lag)
        xy_msd = calc_xy_msd(xy_displacement)
        msd = calc_combined_msd(xy_msd)
        data_dict[time_lag] = msd
    msds = np.array(list(data_dict.values()))
    d = estimate_diffcoff(msds, time_lags)
    np.testing.assert_almost_equal(d, expected_d)


def test_dummy_trajectory(time_traj):
    traj = time_traj
    x_sqdisplacement = calc_sq_displacement_from_zero(traj.x)
    y_sqdisplacement = calc_sq_displacement_from_zero(traj.y)
    msd = calc_combined_msd((x_sqdisplacement, y_sqdisplacement))
    d = estimate_diffcoff(msd, traj.t)
    np.testing.assert_almost_equal(d, expected_d)
