import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union, overload

@dataclass(frozen=True, slots=True)
class Trajectory:
    serialnumber: int
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    species: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.t)
        if not (len(self.x) == len(self.y) == len(self.species) == n):
            raise ValueError("t, x, y, and species must have the same length")
        if self.t.ndim != 1 or self.x.ndim != 1 or self.y.ndim != 1 or self.species.ndim != 1:
            raise ValueError("t, x, y, species must be 1D arrays")
        if not np.issubdtype(self.species.dtype, np.integer):
            raise TypeError("Species must be integer-coded")
        
    def __len__(self) -> int:
        return len(self.t)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        serial_bool = self.serialnumber == other.serialnumber
        t_bool = np.array_equal(self.t, other.t)
        x_bool = np.array_equal(self.x, other.x)
        y_bool = np.array_equal(self.y, other.y)
        species_bool = np.array_equal(self.species, other.species)
        return serial_bool and t_bool and x_bool and y_bool and species_bool
    
    def __getitem__(self, i: int):
        return (
            self.serialnumber,
            self.t[i],
            self.x[i],
            self.y[i],
            self.species[i],
        )

@dataclass(frozen=True, slots=True)
class TrajectorySet:
    trajectories: tuple[Trajectory, ...]

    @classmethod
    def from_list(cls, trajectories: Sequence[Trajectory]) -> "TrajectorySet":
        return cls(tuple(trajectories))
    
    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, key: int):
        return self.trajectories[key]

    @overload
    def __add__(self, other: "TrajectorySet") -> "TrajectorySet": ...
    @overload
    def __add__(self, other: Trajectory) -> "TrajectorySet": ...

    def __add__(self, other: Union["TrajectorySet", Trajectory]) -> "TrajectorySet":
        if isinstance(other, TrajectorySet):
            return TrajectorySet(self.trajectories + other.trajectories)
        if isinstance(other, Trajectory):
            return TrajectorySet(self.trajectories + (other, ))
        return NotImplemented
