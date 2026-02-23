import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Type, Union, overload

import numpy as np


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Immutable container for trajectory."""

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
        self._check_jumps(self.x)
        self._check_jumps(self.y)

    def _check_jumps(self, positions: np.ndarray) -> None:
        jump_sensitivity = 0.5
        max_pos = np.max(np.abs(positions))
        forward_diff = np.diff(positions)
        upper_jumps = forward_diff < jump_sensitivity * max_pos * -1
        lower_jumps = forward_diff > jump_sensitivity * max_pos

        def user_format_warning(
            message: Warning | str,
            category: Type[Warning],
            filename: str,
            lineno: int,
            line: Optional[str] = None,
        ) -> str:
            return f"Warning: {message}\n"

        if (upper_jumps + lower_jumps).sum() != 0:
            warnings.formatwarning = user_format_warning
            warnings.warn(f"Large jumps in trajectory {self.serialnumber} detected.", UserWarning)

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

    def __getitem__(self, i: int) -> tuple[int, float, float, float, int]:
        return (
            self.serialnumber,
            self.t[i],
            self.x[i],
            self.y[i],
            self.species[i],
        )

    @staticmethod
    def adjust_for_periodic_boundaries(
        position: np.ndarray, min_pos: float, max_pos: float
    ) -> np.ndarray:
        size = max_pos - min_pos
        half_delta = 0.5 * (size)
        forward_diff = np.diff(position, prepend=position[0])
        upper_jumps = forward_diff < -1 * half_delta
        lower_jumps = forward_diff > half_delta
        if (upper_jumps + lower_jumps).sum() == 0:
            return position
        upper_jumps_cumsum = upper_jumps.cumsum() * size
        lower_jumps_cumsum = lower_jumps.cumsum() * size * -1
        position_mask = upper_jumps_cumsum + lower_jumps_cumsum
        return position + position_mask


@dataclass(frozen=True, slots=True)
class TrajectorySet:
    """Immutable container for set of trajectories."""

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

    def __getitem__(self, key: int) -> Trajectory:
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
            return TrajectorySet(self.trajectories + (other,))
        return NotImplemented

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over trajectories

        Yields:
            Trajectory: Trajectorie object
        """
        return iter(self.trajectories)

    @property
    def serialnums(self) -> np.ndarray:
        serialnums = np.zeros(len(self))
        for index, traj in enumerate(self):
            serialnums[index] = traj.serialnumber
        return serialnums

    # TODO: Methods .t, .x, ... that return array of values of all trajectories
