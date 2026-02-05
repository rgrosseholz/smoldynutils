from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from smoldynutils.data_objects import Trajectory, TrajectorySet

@dataclass
class SmoldynParser:
    delimiter: str = ","
    dt = 0.5

    def parse_fixed_grid(self, path: str) -> TrajectorySet:
        return parse_smoldyn_molpos_fixed_grid(path, delimiter=self.delimiter)



def parse_smoldyn_molpos_fixed_grid(
        path: str,
        *,
        delimiter: str = ",",
        dtype_xy = np.float32,
        dtype_t = np.float32,
        dtype_species = np.uint16,
        dtype_serialnum = np.uint32
) -> TrajectorySet:
    """Parser based on numpy loadtxt assuming equal size of all trajectories.

    Sorts based on time and serialnumber. Then generates Trajectories based on expected size.

    Args:
        path (str): Path to smoldyn data (assuming listmols2 command)
        delimiter (str, optional): Column delimiter. Defaults to ",".
        dtype_xy (np.float32, optional): xy data type. Defaults to np.float32.
        dtype_t (np.float32, optional): t data type. Defaults to np.float32.
        dtype_species (np.uint16, optional): Species data type. Defaults to np.uint16.

    Returns:
        TrajectorySet: Set of read trajectories.
    """
    file_content = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    if file_content.size == 0:
        raise ValueError("Data file appears to be empty.")
    t = file_content[:, 0].astype(dtype_t, copy=False)
    serial_number = file_content[:, 5].astype(dtype_serialnum, copy=False)
    order = np.lexsort((t, serial_number))
    
    t = t[order]
    serial_number = serial_number[order]
    species = file_content[:, 1].astype(dtype_species, copy=False)[order]
    x = file_content[:, 3].astype(dtype_xy, copy=False)[order]
    y = file_content[:, 4].astype(dtype_xy, copy=False)[order]
    serial_number = file_content[:, 5].astype(dtype_serialnum, copy=False)[order]
    
    serial_ids, serial_start, serial_counts = np.unique(serial_number, return_index=True, return_counts=True)
    
    expected = int(serial_counts[0])
    if not np.all(serial_counts == expected):
        raise NotImplementedError("Not a fixed grid. Serials have different number of timepoints.")

    trajs: list[Trajectory] = []
    for sid, start in zip(serial_ids, serial_start):
        end = start + expected
        trajs.append(
            Trajectory(
                int(sid),
                t=t[start:end],
                x=x[start:end],
                y=y[start:end],
                species=species[start:end],
            )
        )

    return TrajectorySet(tuple(trajs))

