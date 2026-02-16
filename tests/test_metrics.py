import numpy as np
import pytest

from smoldynutils.data_objects import Trajectory
from smoldynutils.metrics import *

expected_x_displacement = 0.2
expected_y_displacement = -0.1


@pytest.fixture
def arrays():
    t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    x = np.array([1.0, 1.2, 1.4], dtype=np.float32)
    y = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    species = np.array([0, 0, 1], dtype=np.uint8)
    return t, x, y, species


@pytest.fixture
def traj(arrays):
    t, x, y, species = arrays
    return Trajectory(1, x=x, y=y, t=t, species=species)


def test_displacement(arrays):
    t, x, y, species = arrays
    x_displ1 = calc_displacements(x, lag=1)
    assert np.isclose(x_displ1, np.array([expected_x_displacement] * 2)).all()
    x_displ2 = calc_displacements(x, lag=2)
    assert np.isclose(x_displ2, np.array([2 * expected_x_displacement])).all()
    with pytest.raises(ValueError):
        calc_displacements(t, lag=3)


def test_xy_displacement(traj):
    x_displ, y_displ = calc_xy_displacement(traj)
    assert np.isclose(x_displ, np.array([expected_x_displacement] * 2)).all()
    assert np.isclose(y_displ, np.array([expected_y_displacement] * 2)).all()

    x_displ, y_displ = calc_xy_displacement(traj, lag=2)
    assert np.isclose(x_displ, np.array([2 * expected_x_displacement])).all()
    assert np.isclose(y_displ, np.array([2 * expected_y_displacement])).all()

    with pytest.raises(ValueError):
        calc_xy_displacement(traj, lag=3)


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
        x_displ, np.array([0, expected_x_displacement, 2 * expected_x_displacement]) ** 2
    ).all()
