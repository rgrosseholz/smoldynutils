import numpy as np
from smoldynutils.data_objects import Trajectory, TrajectorySet
import pytest
from typing import Any, cast

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

def test_raises_errors():
    t, x, y, _ = _get_arrays()
    with pytest.raises(ValueError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([0, 0]))
    with pytest.raises(ValueError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([[0, 0], [1, 1], [2, 2]]))
    with pytest.raises(TypeError):
        traj = Trajectory(serialnumber=1, t=t, x=x, y=y, species=np.array([0, 0, 0.5]))

def test_trajectory_set_init():
    t, x, y, species = _get_arrays()
    traj = Trajectory(1, x, y, t, species)
    trajs = TrajectorySet.from_list([traj]*5)
    assert len(trajs) == 5

    trajs = TrajectorySet.from_list((traj, traj, traj))
    assert len(trajs) == 3
  
def test_trajectory_addition():
    t, x, y, species = _get_arrays()
    traj = Trajectory(1, x, y, t, species)
    trajs = TrajectorySet.from_list([traj]*5)
    added_trajs1 = trajs + traj
    assert len(added_trajs1) == 6
    added_trajs2 = trajs + trajs
    assert len(added_trajs2) == 10
    with pytest.raises(TypeError):
        added_trajs3 = trajs + cast(Any, 5)

