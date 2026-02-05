import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union, overload
import math

@dataclass(frozen=True, slots=True)
class Trajectory:
    """Immutable container for trajectory.
    """
    serialnumber: int
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    species: np.ndarray

    def __post_init__(self) -> None:
        """Performs sensibility checks.

        Raises:
            ValueError: Differing lenghts of t, x, y, or species
            ValueError: >1D for t, x, y, or species
            TypeError: Species is not integer
        """
        n = len(self.t)
        if not (len(self.x) == len(self.y) == len(self.species) == n):
            raise ValueError("t, x, y, and species must have the same length")
        if self.t.ndim != 1 or self.x.ndim != 1 or self.y.ndim != 1 or self.species.ndim != 1:
            raise ValueError("t, x, y, species must be 1D arrays")
        if not np.issubdtype(self.species.dtype, np.integer):
            raise TypeError("Species must be integer-coded")
        
    def __len__(self) -> int:
        """Returns number of points in trajectory

        Returns:
            int: Number of timepoints in trajectory
        """
        return len(self.t)
    
    def __eq__(self, other: object) -> bool:
        """Checks for equality.

        Args:
            other (object): Can be Trajectory or dict containing data.

        Returns:
            bool: True if t, x, y, and species match. False otherwise. NotImplemented if other is not dict or Trajectory.
        """
        if isinstance(other, Trajectory):
            serial_bool = self.serialnumber == other.serialnumber
            t_bool = np.allclose(self.t, other.t)
            x_bool = np.allclose(self.x, other.x)
            y_bool = np.allclose(self.y, other.y)
            species_bool = np.allclose(self.species, other.species)
            return serial_bool and t_bool and x_bool and y_bool and species_bool
        if isinstance(other, dict):
            if len(other["t"]) != len(self):
                return False
            serial_bool = self.serialnumber == other["serialnum"]
            t_bool = np.allclose(self.t, other["t"])
            x_bool = np.allclose(self.x, other["x"])
            y_bool = np.allclose(self.y, other["y"])
            species_bool = np.allclose(self.species, other["species"])
            return serial_bool and t_bool and x_bool and y_bool and species_bool

        return NotImplemented
    
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
    """Immutable container for set of trajectories.

    """
    trajectories: tuple[Trajectory, ...]

    @classmethod
    def from_list(cls, trajectories: Sequence[Trajectory]) -> "TrajectorySet":
        """Create TrajectorySet from sequence of trajectories

        Args:
            trajectories (Sequence[Trajectory]): Sequence of `Trajectory` objects.

        Returns:
            TrajectorySet: Contains provided Trajectories
        """
        return cls(tuple(trajectories))
    
    def __len__(self) -> int:
        """Returns the number of trajectories in the set

        Returns:
            int: Number of stored trajectories
        """
        return len(self.trajectories)

    def __getitem__(self, key: int):
        """Return trajectory by index.

        Args:
            key (int): Index of trajectory to retrieve

        Returns:
            Trajectory: Trajectory at given index
        """
        return self.trajectories[key]

    @overload
    def __add__(self, other: "TrajectorySet") -> "TrajectorySet": ...
    @overload
    def __add__(self, other: Trajectory) -> "TrajectorySet": ...

    def __add__(self, other: Union["TrajectorySet", Trajectory]) -> "TrajectorySet":
        """Combines given trajectories

        Args:
            other (Union[TrajectorySet, Trajectory]): TrajectorySet or Trajectory to combine with current

        Returns:
            TrajectorySet: New TrajectorySet containing the combined trajectories.
        """
        if isinstance(other, TrajectorySet):
            return TrajectorySet(self.trajectories + other.trajectories)
        if isinstance(other, Trajectory):
            return TrajectorySet(self.trajectories + (other, ))
        return NotImplemented
