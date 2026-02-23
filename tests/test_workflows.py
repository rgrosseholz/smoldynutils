import numpy.testing as npt
import pytest

from smoldynutils.data_objects import TrajectorySet
from smoldynutils.workflows import *

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
    x = np.array([0, s, s + 1, s + 1], dtype=np.float64)
    y = np.array([0, 0, -1, s - 1], dtype=np.float64)
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
    x = np.array([0, 1, val_2, val_3], dtype=np.float64)
    y = np.array([0, -1, -val_2, -val_3], dtype=np.float64)
    species = np.array([1, 1, 1, 1], dtype=np.uint8)
    return Trajectory(1, x=x, y=y, t=t, species=species)


@pytest.mark.filterwarnings(r"ignore: Large jumps in trajectory.*:UserWarning")
def test_estimate_timelag_msd(tau_traj):
    lags = [1, 2, 3]
    expected_msd = {lag: 4 * expected_d * lag for lag in lags}
    estimated_msd = estimate_timelag_msd_from_traj(tau_traj, lags)
    for lag in lags:
        npt.assert_almost_equal(estimated_msd[lag], expected_msd[lag])


@pytest.mark.filterwarnings(r"ignore: Large jumps in trajectory.*:UserWarning")
def test_estimate_timelag_diffcoff(tau_traj):
    lags = [1, 2, 3]
    n_trajs = 3
    trajset = TrajectorySet.from_list([tau_traj] * n_trajs)
    estimated_ds = estimate_timelag_diffcoff_from_trajset(trajset, lags)
    summed_ds = np.sum(list(estimated_ds.values()))
    npt.assert_almost_equal(summed_ds, n_trajs * expected_d)
    npt.assert_array_equal(np.array(list(estimated_ds.keys())), np.arange(0, n_trajs))

    serialnums = [1, 2, 3, 99]
    trajs = [
        Trajectory(serialnum, t=tau_traj.t, x=tau_traj.x, y=tau_traj.y, species=tau_traj.species)
        for serialnum in serialnums
    ]
    trajset = TrajectorySet.from_list(trajs)
    estimated_ds = estimate_timelag_diffcoff_from_trajset(trajset, lags)
    npt.assert_array_equal(np.array(list(estimated_ds.keys())), serialnums)


@pytest.mark.filterwarnings(r"ignore: Large jumps in trajectory.*:UserWarning")
def test_estimate_time_msd(time_traj):
    expected_msd = [4 * expected_d * t for t in time_traj.t]
    estimated_msd = estimate_time_msd_from_traj(time_traj)
    for t in time_traj.t:
        npt.assert_almost_equal(estimated_msd, expected_msd)


@pytest.mark.filterwarnings(r"ignore: Large jumps in trajectory.*:UserWarning")
def test_estimate_time_diffcoff(time_traj):
    n_trajs = 3
    trajset = TrajectorySet.from_list([time_traj] * n_trajs)
    estimated_ds = estimate_time_diffcoff_from_trajset(trajset)
    summed_ds = np.sum(list(estimated_ds.values()))
    npt.assert_almost_equal(summed_ds, n_trajs * expected_d)
    npt.assert_array_equal(np.array(list(estimated_ds.keys())), np.arange(0, n_trajs))

    serialnums = [1, 2, 3, 99]
    trajs = [
        Trajectory(serialnum, time_traj.t, time_traj.x, time_traj.y, time_traj.species)
        for serialnum in serialnums
    ]
    trajset = TrajectorySet.from_list(trajs)
    estimated_ds = estimate_time_diffcoff_from_trajset(trajset)
    npt.assert_array_equal(np.array(list(estimated_ds.keys())), serialnums)
