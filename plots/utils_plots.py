"""

Utils functions for plotting in plots_Cloud_base_height_method.ipynb

"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


def add_label(label, ax=None, fig=None, fontsize='medium', va='bottom', x=0., y=1., **kwargs):
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    trans = transforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + trans,
            fontsize=fontsize, va=va, **kwargs)


class SeabornFig2Grid:
    """

    Class from https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot.

    """

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_joint_hist_cbh(labels, preds, cmap, bins, method, label_size, ticks_size, marg_colors, xlabel):
    """
    Plotting joint histogram with marginals distributions for cloud base height predictions.

    :param labels: numpy.array of values for the x-axis (labels).
    :param preds: numpy.array of values for the y-axis (predictions).
    :param cmap: Colormap.
    :param bins: Bins to use for the joint and marginal distributions.
    :param method: Name of the method for the predictions to include in the figure's axis.
    :param label_size: Font size of the axis labels.
    :param ticks_size: Font size of the axis ticks.
    :param marg_colors: List of two colors for the marginal distribution.
    :param xlabel: Label name for the x-axis.
    :return:
    """
    # Jointplot
    plot = sns.jointplot(x=labels, y=preds,
                         height=30,
                         kind='hist',
                         cmap=cmap,
                         cbar=True,
                         joint_kws=dict(bins=bins, kde=True),
                         marginal_kws=dict(bins=bins, fill=False, kde=False),
                         hue_norm=[0, 100]
                         )
    # Set limits
    plot.ax_joint.set_xlim([0, 2750.])
    plot.ax_joint.set_ylim([0, 2750.])
    # Draw bin edges
    for i, bin in enumerate(bins[:-1]):
        plot.ax_joint.axhline(y=bin, xmin=0, xmax=3000., lw=1.8, color='k', alpha=0.5, ls=':')
        plot.ax_joint.axvline(x=bin, ymin=0, ymax=3000., lw=1.8, color='k', alpha=0.5, ls=':')
    # 1:1 blocks
    for i, bin in enumerate(bins[:-1]):
        plot.ax_joint.axhline(y=bin, xmin=bin / 2750, xmax=bins[i + 1] / 2750, lw=5, color='orange', alpha=0.7)
        plot.ax_joint.axhline(y=bins[i + 1], xmin=bin / 2750, xmax=bins[i + 1] / 2750, lw=5, color='orange', alpha=0.7)
        plot.ax_joint.axvline(x=bin, ymin=bin / 2750, ymax=bins[i + 1] / 2750, lw=5, color='orange', alpha=0.7)
        plot.ax_joint.axvline(x=bins[i + 1], ymin=bin / 2750, ymax=bins[i + 1] / 2750, lw=5, color='orange', alpha=0.7)
    # Setup axes of jointplot (ticks and labels)
    plot.fig.axes[0].tick_params(color='white')
    plot.ax_joint.set_ylabel('{} cloud base height ($m$)'.format(method), fontsize=label_size)
    plot.ax_joint.set_yticks(np.arange(0., 2750., 250.),
                             [str(int(a)) for a in np.arange(0., 2750., 250.)],
                             fontsize=ticks_size)
    plot.ax_joint.set_xlabel('{} cloud base height ($m$)'.format(xlabel), fontsize=label_size)
    plot.ax_joint.set_xticks(np.arange(0., 2750., 250.),
                             [str(int(a)) for a in np.arange(0., 2750., 250.)],
                             fontsize=ticks_size)
    # Setup marginals
    for patch in plot.ax_marg_x.patches:
        patch.set(facecolor=marg_colors[0], alpha=0.5, edgecolor='white')
    for patch in plot.ax_marg_y.patches:
        patch.set(facecolor=marg_colors[1], alpha=0.5, edgecolor='white')
    plot.ax_marg_x.tick_params(color='white')
    plot.ax_marg_y.tick_params(color='white')
    # Adjust plot
    plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
    # Setup axes for colorbar
    pos_joint_ax = plot.ax_joint.get_position()
    pos_marg_x_ax = plot.ax_marg_x.get_position()
    plot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    plot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # Setup colorbar ticks and labels
    cbar_ticks = plot.fig.axes[-1].get_yticks()
    plot.fig.axes[-1].set_yticklabels([int(t) for t in cbar_ticks], fontsize=ticks_size)
    plot.fig.axes[-1].tick_params(axis='y', color='white')
    for axis in ['left', 'bottom']:
        plot.ax_joint.spines[axis].set_linewidth(0)
        plot.ax_marg_x.spines[axis].set_linewidth(0)
        plot.ax_marg_y.spines[axis].set_linewidth(0)
    for axis in ['right', 'top']:
        plot.ax_joint.spines[axis].set_linewidth(0)
    return plot


def bin_value_cat(val, bins):
    """
    Bin values into categories.
    :param val: xr.DataArray (lat, lon)
    :param bins:
    :return: binned_val
    """
    return xr.apply_ufunc(
        np.digitize, val, kwargs={'bins': bins, 'right': False}
    )
