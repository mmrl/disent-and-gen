"""
Code used to plot Hinton matrices of the regression coefficients learned
using the disentanglment evaluation framewwork of Eastwood & Williams, 2018.

Taken from:
    https://github.com/cianeastwood/qedr/blob/master/lib/eval/hinton.py

Based on:
1) https://github.com/tonysyu/mpltools/blob/master/mpltools/special/hinton.py
2) http://tonysyu.github.io/mpltools/auto_examples/special/plot_hinton.html
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import transforms
from matplotlib import ticker

__all__ = ['hinton']


class SquareCollection(collections.RegularPolyCollection):
    """Return a collection of squares."""

    def __init__(self, **kwargs):
        super(SquareCollection, self).__init__(4, rotation=np.pi/4., **kwargs)

    def get_transform(self):
        """Return transform scaling circle areas to data space."""
        ax = self.axes
        pts2pixels = 72.0 / ax.figure.dpi
        scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
        scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
        return transforms.Affine2D().scale(scale_x, scale_y)


def hinton(inarray, x_label=None, y_label=None, max_value=None,
           use_default_ticks=True, ax=None, fontsize=14):
    """Plot Hinton diagram for visualizing the values of a 2D array.
    Plot representation of an array with positive and negative values
    represented by white and black squares, respectively. The size of each
    square represents the magnitude of each value.
    Unlike the hinton demo in the matplotlib gallery [1]_, this implementation
    uses a RegularPolyCollection to draw squares, which is much more efficient
    than drawing individual Rectangles.
    .. note::
        This function inverts the y-axis to match the origin for arrays.
    .. [1] http://matplotlib.sourceforge.net/examples/api/hinton_demo.html
    Parameters
    ----------
    inarray : array
        Array to plot.
    max_value : float
        Any *absolute* value larger than `max_value` will be represented by a
        unit square.
    use_default_ticks: boolean
        Disable tick-generation and generate them outside this function.
    """

    ax = ax if ax is not None else plt.gca()
    ax.set_facecolor('gray')
    # make sure we're working with a numpy array, not a numpy matrix
    inarray = np.asarray(inarray)
    height, width = inarray.shape
    if max_value is None:
        max_value = 2**np.ceil(np.log(np.max(np.abs(inarray)))/np.log(2))
    values = np.clip(inarray/max_value, -1, 1)
    rows, cols = np.mgrid[:height, :width]

    pos = np.where(values > 0)
    neg = np.where(values < 0)
    for idx, color in zip([pos, neg], ['white', 'black']):
        if len(idx[0]) > 0:
            xy = list(zip(cols[idx], rows[idx]))
            circle_areas = np.pi / 2 * np.abs(values[idx])
            squares = SquareCollection(sizes=circle_areas,
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor=color, edgecolor=color)
            ax.add_collection(squares, autolim=True)

    ax.axis('scaled')
    # set data limits instead of using xlim, ylim.
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(height-0.5, -0.5)
    ax.grid(False)
    ax.tick_params(direction='in', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize)

    if use_default_ticks:
        ax.xaxis.set_major_locator(IndexLocator())
        ax.yaxis.set_major_locator(IndexLocator())


class IndexLocator(ticker.Locator):
    def __init__(self, max_ticks=10):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))


def plot_hinton_matrices(R_matrices, model_names, factors, fig=None):
    """
    Convenience method to plot the hinton matrices of a set of regression
    coefficients.
    """
    if fig is None:
        fig, axes = plt.subplots(1, len(model_names), figsize=(20, 5))
    else:
        axes = fig.axes

    latent_size = R_matrices[0].shape[0]

    for i in range(len(model_names)):
        ylabel = 'latent' if i == 0 else ''

        hinton(R_matrices[i], 'factor', ylabel,
               ax=axes[i], fontsize=18,
               use_default_ticks=False)

        axes[i].set_xticks(range(len(factors)))
        axes[i].set_xticklabels(factors)
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(90)

        if i == 0:
            axes[i].set_yticks(range(latent_size))
        else:
            axes[i].set_yticks([])

        axes[i].set_title('{0}'.format(model_names[i]), fontsize=20)

    return fig
