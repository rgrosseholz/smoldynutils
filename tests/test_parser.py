import pytest
import numpy as np
from smoldynutils.parsing import SmoldynParser

sample_smoldyn_file = (
    "1,1,0,12.7828,2.06726,100\n"
    "1,1,0,1.01686,8.12141,99\n"
    "2,1,0,12.8000,2.10000,100\n"
    "2,1,0,1.02000,8.13000,99\n"
)

sample_smoldyn_file_ragged = (
    "1,1,0,12.7828,2.06726,100\n"
    "1,1,0,1.01686,8.12141,99\n"
    "2,1,0,12.8000,2.10000,100\n"
    "2,1,0,1.02000,8.13000,99\n"
    "1,1,0,12.7828,2.06726,100\n"
)

empty_file = ""


def _write_sample(tmp_path):
    file = tmp_path / "sample.csv"
    file.write_text(sample_smoldyn_file)
    return file


def _write_sample_ragged(tmp_path):
    file = tmp_path / "sample2.csv"
    file.write_text(sample_smoldyn_file_ragged)
    return file


def _write_empty(tmp_path):
    file = tmp_path / "sample2.csv"
    file.write_text(empty_file)
    return file


def test_fixed_grid_parser(tmp_path):
    path = _write_sample(tmp_path)
    parser = SmoldynParser()
    ts = parser.parse_fixed_grid(str(path))
    assert len(ts) == 2
    assert ts[0] == {
        "serialnum": 99,
        "t": np.array([1, 2]),
        "x": np.array([1.01686, 1.02]),
        "y": np.array([8.12141, 8.13]),
        "species": np.array([1, 1]),
    }

    path = _write_sample_ragged(tmp_path)
    parser = SmoldynParser()
    with pytest.raises(NotImplementedError):
        ts = parser.parse_fixed_grid(str(path))

    path = _write_empty(tmp_path)
    parser = SmoldynParser()
    with pytest.warns(UserWarning, match="input contained no data"):
        with pytest.raises(ValueError):
            ts = parser.parse_fixed_grid(str(path))
