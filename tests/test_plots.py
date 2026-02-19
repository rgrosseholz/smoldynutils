import matplotlib.pyplot as plt
import numpy as np
import pytest

from smoldynutils.plots import *


def test_plot_gauss_comparison():
    disp = np.random.randn(10)
    gauss = np.random.randn(10)
    fig, ax = plt.subplots()
    initial_patches = len(ax.patches)
    returned = plot_gauss_comparison(disp, gauss, ax, title="Testtitle")
    assert returned is ax
    assert len(returned.patches) > initial_patches
    assert returned.get_title() == "Testtitle"
