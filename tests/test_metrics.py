import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from smoldynutils.data_objects import Trajectory
from smoldynutils.metrics import *

expected_x_displacement = 1
expected_y_displacement = -1
expected_msd = expected_x_displacement**2 + expected_y_displacement**2
expected_D = 0.5

# TODO: I made some design mistakes for the tests here.
# Crucially, unit tests should not built on top of each other, but rather function
# independently.


@pytest.fixture
def arrays():
    t = np.array([0, 1, 2, 3], dtype=np.float32)
    x = np.array([0, 1, 2, 3], dtype=np.float32)
    y = np.array([0, -1, -2, -3], dtype=np.float32)
    species = np.array([1, 1, 1, 1], dtype=np.uint8)
    return t, x, y, species


@pytest.fixture
def traj(arrays):
    t, x, y, species = arrays
    return Trajectory(1, x=x, y=y, t=t, species=species)


@pytest.fixture
def unmoving_traj():
    t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    y = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    species = np.array([1, 1, 1], dtype=np.uint8)
    return Trajectory(1, t, x, y, species)


def test_displacement(arrays):
    t, x, y, species = arrays
    x_displ1 = calc_displacements(x, lag=1)
    assert np.isclose(x_displ1, np.array([expected_x_displacement] * (len(x) - 1))).all()
    x_displ2 = calc_displacements(x, lag=2)
    assert np.isclose(x_displ2, np.array([2 * expected_x_displacement] * (len(x) - 2))).all()
    with pytest.raises(ValueError):
        calc_displacements(t, lag=4)


def test_xy_displacement(traj, unmoving_traj):
    x_displ, y_displ = calc_xy_displacement(traj)
    assert np.isclose(x_displ, np.array([expected_x_displacement] * (len(traj.x) - 1))).all()
    assert np.isclose(y_displ, np.array([expected_y_displacement] * (len(traj.y) - 1))).all()

    x_displ, y_displ = calc_xy_displacement(traj, lag=2)
    assert np.isclose(x_displ, np.array([2 * expected_x_displacement] * (len(traj.x) - 2))).all()
    assert np.isclose(y_displ, np.array([2 * expected_y_displacement] * (len(traj.x) - 2))).all()

    with pytest.raises(ValueError, match="Timelag is bigger than number of datapoints in x or y"):
        calc_xy_displacement(traj, lag=4)

    x_displ, y_displ = calc_xy_displacement(unmoving_traj)
    np.testing.assert_allclose(x_displ, 0)
    np.testing.assert_allclose(y_displ, 0)


def test_msd(traj):
    displacement = calc_displacements(traj.x)
    msd = calc_msd(displacement)
    assert np.isclose(msd, expected_x_displacement**2)
    displacement2 = calc_displacements(traj.x, lag=2)
    msd = calc_msd(displacement2)
    assert np.isclose(msd, (2 * expected_x_displacement) ** 2)


def test_xy_msd(traj):
    xy_displ = calc_xy_displacement(traj)
    x_msd, y_msd = calc_xy_msd(xy_displ)
    assert np.isclose(x_msd, expected_x_displacement**2)
    assert np.isclose(y_msd, expected_y_displacement**2)


def test_displacement_from_zero(traj):
    x_displ = calc_sq_displacement_from_zero(traj.x)
    assert np.isclose(
        x_displ,
        np.array(
            [0, expected_x_displacement, 2 * expected_x_displacement, 3 * expected_x_displacement]
        )
        ** 2,
    ).all()


def test_calc_combined_msd(traj, unmoving_traj):
    xy_displ = calc_xy_displacement(traj)
    x_msd, y_msd = calc_xy_msd(xy_displ)
    np.testing.assert_almost_equal(calc_combined_msd((x_msd, y_msd)), expected_msd)
    xy_displ = calc_xy_displacement(unmoving_traj)
    x_msd, y_msd = calc_xy_msd(xy_displ)
    np.testing.assert_allclose(calc_combined_msd((x_msd, y_msd)), np.array(0))


def test_estimate_diffcoff(traj, unmoving_traj):
    xy_disp = calc_xy_displacement(unmoving_traj)
    x_msd, y_msd = calc_xy_msd(xy_disp)
    msd = calc_combined_msd((x_msd, y_msd))
    with pytest.warns(OptimizeWarning):
        d = estimate_diffcoff(msd, np.array([1]))
        np.testing.assert_almost_equal(d, 0)
    xy_disp = calc_xy_displacement(traj)
    msds = calc_xy_msd(xy_disp)
    msd = calc_combined_msd(msds)
    with pytest.warns(OptimizeWarning):
        np.testing.assert_almost_equal(estimate_diffcoff(msd, np.array([1])), expected_D)
    with pytest.warns(UserWarning):
        estimate_diffcoff(msd, np.array([1]), add_epsilon=True)

    np.testing.assert_equal(
        estimate_diffcoff(np.array([0, 1, 2]), np.array([1, 2, 3]), add_epsilon=True), 0.25
    )


def test_estimate_diffcoff_full_return():
    msd = np.zeros((1))
    with pytest.warns(OptimizeWarning):
        full_d = estimate_diffcoff_fullinfo(msd, np.array([1]))
        assert len(full_d) == 2
