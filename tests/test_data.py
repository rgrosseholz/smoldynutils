import warnings
from typing import Any, cast

import numpy as np
import pytest

from smoldynutils.data_objects import Trajectory, TrajectorySet


def _get_arrays():
    t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    x = np.array([1.0, 1.2, 1.4], dtype=np.float32)
    y = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    species = np.array([0, 0, 1], dtype=np.uint8)
    return t, x, y, species


def test_traj_construction():
    t, x, y, species = _get_arrays()
    traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=species)
    assert traj.t.shape == t.shape
    assert traj.x.shape == x.shape
    assert traj.y.shape == y.shape
    assert traj.species.shape == species.shape
    assert len(traj) == 3
    assert traj == traj
    assert traj == {"t": t, "x": x, "y": y, "species": species, "serialnum": 1}
    assert traj != {"t": np.array([0.0, 1]), "x": x, "y": y, "species": species, "serialnum": 1}
    assert (traj.__eq__(1)) is NotImplemented
    assert traj[0] == (1, t[0], x[0], y[0], species[0])


def test_raises_errors():
    t, x, y, species = _get_arrays()
    with pytest.raises(ValueError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([0, 0]))
    with pytest.raises(ValueError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([[0, 0], [1, 1], [2, 2]]))
    with pytest.raises(TypeError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([0, 0, 0.5]))


def test_trajectory_set_init():
    t, x, y, species = _get_arrays()
    traj = Trajectory(1, t, x, y, species)
    trajs = TrajectorySet.from_list([traj] * 5)
    assert len(trajs) == 5

    trajs = TrajectorySet.from_list((traj, traj, traj))
    assert len(trajs) == 3


def test_trajectory_addition():
    t, x, y, species = _get_arrays()
    traj = Trajectory(1, t, x, y, species)
    trajs = TrajectorySet.from_list([traj] * 5)
    added_trajs1 = trajs + traj
    assert len(added_trajs1) == 6
    added_trajs2 = trajs + trajs
    assert len(added_trajs2) == 10
    with pytest.raises(TypeError):
        added_trajs3 = trajs + cast(Any, 5)


def test_trajectory_iter():
    t, x, y, species = _get_arrays()
    traj = Trajectory(1, t, x, y, species)
    trajs = TrajectorySet.from_list([traj] * 5)
    trajs = trajs + Trajectory(2, t, x, y, species)
    for i, traj in enumerate(trajs):
        if i + 1 < len(trajs):
            assert traj.serialnumber == 1
        else:
            assert traj.serialnumber == 2


def test_raises_jump_warning():
    t, _, y, species = _get_arrays()
    x = np.array([0, 2, 4, 0])
    t = np.append(t, 4)
    y = np.append(y, 0.2)
    species = np.append(species, 1)
    with pytest.warns(UserWarning, match="Large jumps in trajectory 1 detected."):
        traj = Trajectory(1, t, x, y, species)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # traj = Trajectory(1, t, x, y, species)
        msg = warnings.formatwarning("hello", UserWarning, "file.py", 123)
        assert msg == "Warning: hello\n"


def test_trajset_serialnums():
    t, x, y, species = _get_arrays()
    traj_list = [Trajectory(n, t, x, y, species) for n in range(1, 5)]
    trajs = TrajectorySet.from_list(traj_list)
    np.testing.assert_equal(
        trajs.serialnums,
        [
            1,
            2,
            3,
            4,
        ],
    )
