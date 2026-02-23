import matplotlib.pyplot as plt
import numpy as np
import pytest

from smoldynutils.data_objects import Trajectory, TrajectorySet
from smoldynutils.plots import *


@pytest.fixture
def plot():
    fig, ax = plt.subplots()
    return fig, ax


def test_plot_gauss_comparison(plot):
    disp = np.random.randn(10)
    gauss = np.random.randn(10)
    fig, ax = plot
    initial_patches = len(ax.patches)
    returned = plot_gauss_comparison(disp, gauss, ax, title="Testtitle")
    assert returned is ax
    assert len(returned.patches) > initial_patches
    assert returned.get_title() == "Testtitle"


def test_plot_trajectorie(plot):
    fig, ax = plot
    initial_lines = len(ax.patches)
    initial_dots = len(ax.collections)
    vals = np.linspace(0, 10, 11)
    species = np.array([1] * 11)
    traj = Trajectory(1, vals, vals, vals, species)
    returned = plot_trajectorie(traj, ax, "test")
    assert returned is ax
    assert len(returned.lines) > initial_lines
    assert len(returned.collections) > initial_dots
    assert returned.get_title() == "test"


def test_plot_trajectories(plot):
    fig, ax = plot
    initial_lines = len(ax.patches)
    initial_dots = len(ax.collections)
    vals = np.linspace(0, 10, 11)
    species = np.array([1] * 11)
    traj = Trajectory(1, vals, vals, vals, species)
    trajset = TrajectorySet.from_list([traj] * 5)
    returned = plot_trajectories(trajset, ax, "test")
    assert returned is ax
    assert len(returned.lines) > initial_lines
    assert len(returned.collections) > initial_dots
    assert returned.get_title() == "test"


def test_plot_msd(plot):
    fig, ax = plot
    initial_lines = len(ax.lines)
    vals = np.linspace(0, 10, 11)
    returned = plot_msd(msd=vals, ax=ax, time=vals, title="test")
    one_msd_lines = len(returned.lines)
    assert returned is ax
    assert one_msd_lines > initial_lines
    assert returned.get_title() == "test"

    fig, ax = plot
    multi_msd = np.array([vals] * 5)
    returned = plot_msd(msd=multi_msd, ax=ax, time=vals, title="test")
    assert returned is ax
    assert len(returned.lines) > one_msd_lines
    assert returned.get_title() == "test"

    plot_msd(msd=multi_msd, ax=ax)

    with pytest.raises(ValueError, match="Input MSD array is > 2D"):
        array_3d = np.zeros((2, 2, 2))
        plot_msd(array_3d, ax)

    with pytest.raises(ValueError, match="Input MSD array and time array have no shape in common."):
        array_no_ax_in_common = np.zeros((5, 7))
        plot_msd(array_no_ax_in_common, ax, time=vals)


def test_plot_msd_compare(plot):
    fig, ax = plot
    vals = np.linspace(0, 10, 11)
    msd_ax = plot_msd(vals, ax, time=vals)
    msd_lines = len(msd_ax.lines)
    fig, ax = plot
    initial_lines = len(ax.lines)
    returned = plot_msd_comparison(msd=vals, theoretical_msd=vals, ax=ax, time=vals, title="test")
    assert returned is ax
    assert len(returned.lines) > initial_lines
    assert len(returned.lines) > msd_lines
    assert returned.get_title() == "test"

    plot_msd_comparison(vals, vals, ax)

    with pytest.raises(ValueError):
        plot_msd_comparison(vals, np.linspace(0, 10, 5), ax)


def test_plot_diffconst_hist(plot):
    fig, ax = plot
    initial_patches = len(ax.patches)
    vals = np.array([0, 0, 0, 1, -1])
    reference_d = 0
    returned = plot_diffconst_hist(
        diffcoffs=vals, reference_diffcoff=reference_d, ax=ax, title="test"
    )
    assert returned is ax
    assert len(returned.patches) > initial_patches
    assert returned.get_title() == "test"
