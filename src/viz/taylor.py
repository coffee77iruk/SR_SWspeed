"""
Taylor diagram (Taylor 2001) for comparing model standard deviation and
correlation against OMNI in a single polar plot.

plot_taylor_diagram -> entire-test-period diagram for a set of model columns,
                       filtered the same way as evaluate_metrics(df, group_by=None)
                       (Oct-Dec test months, ICME periods excluded).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa


class TaylorDiagram(object):
    """Taylor diagram plotting utility: standard deviation (radius) vs.
    correlation coefficient (angle) against a single reference point."""

    def __init__(self, STD, fig=None, rect=111, label='_'):
        self.STD = STD
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.6 * self.STD
        gh = fa.GridHelperCurveLinear(tr, extremes=(0, (np.pi / 2), self.smin, self.smax),
                                      grid_locator1=gl1, tick_formatter1=tf1)
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)
        # Angle axis
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].label.set_text("Correlation coefficient")
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        # X axis
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_text("Standard deviation [km/s]")
        ax.axis['left'].toggle(ticklabels=True, label=True)
        ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
        ax.axis['left'].label.set_axis_direction('bottom')
        # Y axis
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].label.set_text("Standard deviation [km/s]")
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        ax.axis['right'].label.set_axis_direction('top')
        # Useless
        ax.axis['bottom'].set_visible(False)
        # Contours along standard deviations
        ax.grid()
        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates
        # Add reference point and STD contour
        l, = self.ax.plot([0], self.STD, 'k*', ls='', ms=15, label=label)
        t = np.linspace(0, (np.pi / 2.0))
        r = np.zeros_like(t) + self.STD
        self.ax.plot(t, r, 'k--', label='_')
        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, STD, r, *args, **kwargs):
        l, = self.ax.plot(np.arccos(r), STD, *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)
        return l

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
        RMSE = np.sqrt(np.power(self.STD, 2) + np.power(rs, 2) - (2.0 * self.STD * rs * np.cos(ts)))
        contours = self.ax.contour(ts, rs, RMSE, levels, **kwargs)
        return contours


def plot_taylor_diagram(df, model_specs, target_col="speed", test_months=(10, 11, 12), save_path=None):
    """
    Taylor diagram over the entire test period. model_specs is a list of
    (column, label, color, marker) tuples. Filtering matches
    evaluate_metrics(df, group_by=None): Oct-Dec test months, ICME periods
    excluded. All models are compared against a common reference subset
    (rows where every model column and target_col is present), so the
    reference OMNI point is a single fixed standard deviation shared by
    every sample rather than recomputed per model.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[df["datetime"].dt.month.isin(test_months)]
    if "is_ICME" in df.columns:
        df = df[~df["is_ICME"]]

    cols = [col for col, *_ in model_specs]
    sub = df[[target_col] + cols].dropna()
    y_true = sub[target_col].values
    obs_std = float(np.std(y_true))

    fig = plt.figure(figsize=(9, 8))
    dia = TaylorDiagram(obs_std, fig=fig, rect=111, label="OMNI")
    plt.clabel(dia.add_contours(colors="#808080"), inline=1, fontsize=10)

    for col, label, color, marker in model_specs:
        y_pred = sub[col].values
        pred_std = float(np.std(y_pred))
        cc = float(np.corrcoef(y_true, y_pred)[0, 1])
        dia.add_sample(pred_std, cc, label=label, marker=marker,
                       mec=color, mfc="none", mew=2.0, ms=12)

    spl = [p.get_label() for p in dia.samplePoints]
    legend = fig.legend(dia.samplePoints, spl, numpoints=1, loc="upper right",
                        prop=dict(size=11), frameon=True, handlelength=2, handleheight=2)
    for handle in legend.legend_handles:
        handle.set_linewidth(0)

    plt.title("Taylor Diagram (entire test period)", fontsize=18, pad=30)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
