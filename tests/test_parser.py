import numpy as np
import pytest

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

sample_smoldyn_file_borderjump = (
    "1,1,0,12.7828,2.06726,100\n"
    "1,1,0,1.01686,8.12141,99\n"
    "2,1,0,12.8000,2.10000,100\n"
    "2,1,0,1.02000,8.13000,99\n"
    "3,1,0,0.10000,2.10000,100\n"
    "3,1,0,1.02000,8.13000,99\n"
)


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


def _write_borderjump(tmp_path):
    file = tmp_path / "sample3.csv"
    file.write_text(sample_smoldyn_file_borderjump)
    return file


def test_fixed_grid_parser(tmp_path):
    path = _write_sample(tmp_path)
    parser = SmoldynParser(str(path))
    ts = parser.parse_fixed_grid()
    assert len(ts) == 2
    assert ts[0] == {
        "serialnum": 99,
        "t": np.array([1, 2]),
        "x": np.array([1.01686, 1.02]),
        "y": np.array([8.12141, 8.13]),
        "species": np.array([1, 1]),
    }

    path = _write_sample_ragged(tmp_path)
    parser = SmoldynParser(str(path))
    with pytest.raises(NotImplementedError):
        ts = parser.parse_fixed_grid()

    path = _write_empty(tmp_path)
    parser = SmoldynParser(str(path))
    with pytest.warns(UserWarning, match="input contained no data"):
        with pytest.raises(ValueError):
            ts = parser.parse_fixed_grid()

    path = _write_borderjump(tmp_path)
    parser = SmoldynParser(str(path), min_val=0, max_val=12.8)
    ts = parser.parse_fixed_grid()
    assert len(ts) == 2
    print(ts[1].x)
    assert ts[1] == {
        "serialnum": 100,
        "t": np.array([1, 2, 3]),
        "x": np.array([12.7828, 12.8000, 12.9000]),
        "y": np.array([2.06726, 2.10000, 2.10000]),
        "species": np.array([1, 1, 1]),
    }
