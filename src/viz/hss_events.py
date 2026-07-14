"""
Visualization for HSS/SIR event detection: solar wind speed profiles (full
period stacked by year, and per-Carrington-Rotation zoom), event-detection
confusion matrix, and peak timing/speed bias violin plots.
"""

import os
import re
import glob
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from utils.hse_detection import detect_HSE_blocks, plot_peaks, shade_icme, get_cr_date_range
from utils.icme import make_icme_mask
from viz.ch_parameters import (
    load_and_calibrate, get_spoca_ch_union, get_rgba_and_extent, set_hpc_axes, draw_ch_contour,
)

MODEL_COLORS = {
    "OMNI": "black", "ESWF": "green", "WSA-ENLIL": "deepskyblue",
    "Persistence (27 days)": "orange", "SR-derived formula": "red",
    "Average prediction (Collin+2025)": "gray",
}


def _shade_event_windows(ax, omni_peaks, time_window_hr=24, speed_window=100,
                         facecolor="cyan", alpha=0.2, edgecolor="cyan", lw=1.5):
    """Rectangle around each OMNI peak showing the time x speed acceptance
    window used by match_peaks_with_window()."""
    dt_half = timedelta(hours=time_window_hr)
    for t_peak, v_peak in omni_peaks:
        x_left = mdates.date2num(t_peak - dt_half)
        x_right = mdates.date2num(t_peak + dt_half)
        rect = mpatches.Rectangle(
            (x_left, v_peak - speed_window), x_right - x_left, 2 * speed_window,
            linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
            linestyle="--", alpha=alpha, zorder=2,
        )
        ax.add_patch(rect)


def plot_speed_profile_by_year(df, series_specs, icme_intervals, years=range(2010, 2025),
                               test_months=(10, 11, 12), time_col="datetime",
                               show_event_windows=False, event_time_window_hr=24,
                               event_speed_window=100, save_path=None):
    """
    One row per year, Oct-Dec speed profile for every (colname, label) in
    series_specs plus the observed "OMNI" series, with ICME shading, detected
    HSS peaks marked, and (if the formula's uncertainty band columns are
    present) a shaded band around the SR-derived formula.

    series_specs : list of (colname, label) tuples. "speed" (OMNI) and any
                   label containing "SR" get a solid line + linewidth 3;
                   others get a dashed line + linewidth 2. Colors come from
                   MODEL_COLORS.
    show_event_windows : overlay the time x speed acceptance-window rectangles
                   around each OMNI peak (see match_peaks_with_window).

    Returns (fig, all_peak_lists) where all_peak_lists[label] is the list of
    (timestamp, speed) peaks detected for that series across all years shown
    -- reusable directly by evaluate_event_detection().
    """
    time_all = df[time_col]
    icme_mask = make_icme_mask(time_all, icme_intervals)
    all_peak_lists = {label: [] for _, label in series_specs}
    all_peak_lists.setdefault("OMNI", [])

    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(30, 4 * n_years), sharex=False, sharey=True)
    if n_years == 1:
        axes = [axes]

    has_band = {"max_sqrt_AP", "min_sqrt_AP"}.issubset(df.columns)
    sr_col = next((c for c, l in series_specs if "SR" in l), None)

    # Detect once per series over the full range, not per year.
    sir_dicts = {"OMNI": detect_HSE_blocks(time_all, df["speed"])}
    for colname, label in series_specs:
        sir_dicts[label] = detect_HSE_blocks(time_all, df[colname])

    for ax, year in zip(axes, years):
        year_df = df[(time_all.dt.year == year) & (time_all.dt.month.isin(test_months))]
        if year_df.empty:
            ax.set_visible(False)
            continue
        time_year = year_df[time_col]

        shade_icme(ax, icme_intervals, t0=time_year.iloc[0], t1=time_year.iloc[-1],
                  facecolor="gray", alpha=0.4)

        if has_band and sr_col:
            ax.fill_between(year_df[time_col],
                           year_df[sr_col] - df.loc[year_df.index, "min_sqrt_AP"],
                           year_df[sr_col] + df.loc[year_df.index, "max_sqrt_AP"],
                           color="red", alpha=0.2)

        ax.plot(year_df[time_col], year_df["speed"], color=MODEL_COLORS["OMNI"], lw=3, label="OMNI")
        omni_peaks = plot_peaks(ax, sir_dicts["OMNI"], df["speed"], time_all, time_year, icme_mask,
                                color=MODEL_COLORS["OMNI"], markersize=12)
        all_peak_lists["OMNI"].extend(omni_peaks)

        for colname, label in series_specs:
            color = MODEL_COLORS.get(label, "gray")
            ls, lw = ("-", 3) if "SR" in label else ("--", 2)
            ax.plot(year_df[time_col], year_df[colname], color=color, lw=lw, linestyle=ls, label=label)

            peaks = plot_peaks(ax, sir_dicts[label], df[colname], time_all, time_year, icme_mask,
                               color=color, markersize=12)
            all_peak_lists[label].extend(peaks)

        if show_event_windows:
            _shade_event_windows(ax, omni_peaks, time_window_hr=event_time_window_hr,
                                 speed_window=event_speed_window)

        ax.set_ylim(250, 900)
        ax.set_yticks([400, 600, 800])
        ax.tick_params(axis="y", labelsize=24)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax.tick_params(axis="x", labelsize=24)
        ax.set_ylabel("Speed [km/s]", fontsize=28, labelpad=15)
        ax.margins(x=0.001)

    axes[-1].set_xlabel("Date [month-year]", fontsize=28, labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=28,
              frameon=False, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=400)
    return fig, all_peak_lists


def plot_speed_profile_cr(df, cr_df, cr_pairs, series_specs, icme_intervals,
                          time_col="datetime", print_metrics=False, save_path=None):
    """
    One row per (cr_start, cr_end) pair in cr_pairs, zoomed to that Carrington
    Rotation window, with ICME shading, the SR uncertainty band, and detected
    HSS peaks marked for every series in series_specs (+ observed OMNI).

    print_metrics : also print MAE/RMSE/CC per model within each CR window
                   (matches the sanity-check printout from the original
                   exploration notebook).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    time_all = df[time_col]
    icme_mask = make_icme_mask(time_all, icme_intervals)
    has_band = {"max_sqrt_AP", "min_sqrt_AP"}.issubset(df.columns)
    sr_col = next((c for c, l in series_specs if "SR" in l), None)

    n_rows = len(cr_pairs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(30, 5 * n_rows), sharex=False, sharey=True)
    if n_rows == 1:
        axes = [axes]

    sir_dicts = {"OMNI": detect_HSE_blocks(time_all, df["speed"])}
    for colname, label in series_specs:
        sir_dicts[label] = detect_HSE_blocks(time_all, df[colname])

    for ax, target_crs in zip(axes, cr_pairs):
        t_starts, t_ends = [], []
        for cr_num in target_crs:
            t_start, t_end = get_cr_date_range(cr_df, cr_num)
            t_starts.append(t_start)
            t_ends.append(t_end)
        t_plot_start, t_plot_end = min(t_starts), max(t_ends)

        cr_df_plot = df[(time_all >= t_plot_start) & (time_all < t_plot_end)]
        if cr_df_plot.empty:
            ax.set_visible(False)
            continue
        time_cr = cr_df_plot[time_col]

        if print_metrics:
            obs = cr_df_plot["speed"]
            print(f"\n=== CR {target_crs[0]}-{target_crs[1]} Metrics ===")
            print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'CC':>8}")
            print("-" * 55)
            for colname, label in series_specs:
                pred = cr_df_plot[colname]
                mask = obs.notna() & pred.notna()
                o, p = obs[mask].to_numpy(), pred[mask].to_numpy()
                mae = mean_absolute_error(o, p)
                rmse = np.sqrt(mean_squared_error(o, p))
                cc = np.corrcoef(o, p)[0, 1]
                print(f"{label:<25} {mae:>8.1f} {rmse:>8.1f} {cc:>8.3f}")
            print("-" * 55)

        shade_icme(ax, icme_intervals, t0=t_plot_start, t1=t_plot_end, facecolor="gray", alpha=0.4)
        ax.axvline(t_ends[0], color="gray", linestyle=":", linewidth=2, alpha=0.8)
        for cr_num, t_s in zip(target_crs, t_starts):
            ax.text(t_s + timedelta(hours=6), 868, f"CR {cr_num}", ha="left", va="top",
                   fontsize=24, color="gray")

        if has_band and sr_col:
            ax.fill_between(cr_df_plot[time_col],
                           cr_df_plot[sr_col] - df.loc[cr_df_plot.index, "min_sqrt_AP"],
                           cr_df_plot[sr_col] + df.loc[cr_df_plot.index, "max_sqrt_AP"],
                           color="red", alpha=0.2)

        ax.plot(cr_df_plot[time_col], cr_df_plot["speed"], color=MODEL_COLORS["OMNI"], lw=3, label="OMNI")
        plot_peaks(ax, sir_dicts["OMNI"], df["speed"], time_all, time_cr, icme_mask,
                  color=MODEL_COLORS["OMNI"], markersize=12)

        for colname, label in series_specs:
            color = MODEL_COLORS.get(label, "gray")
            ls, lw = ("-", 3) if "SR" in label else ("--", 2)
            ax.plot(cr_df_plot[time_col], cr_df_plot[colname], color=color, lw=lw, linestyle=ls, label=label)
            plot_peaks(ax, sir_dicts[label], df[colname], time_all, time_cr, icme_mask, color=color, markersize=12)

        ax.set_xlim(t_plot_start, t_plot_end)
        ax.set_ylim(250, 900)
        ax.set_yticks([400, 600, 800])
        ax.tick_params(axis="y", labelsize=26)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelsize=26)
        ax.set_ylabel("Speed [km/s]", fontsize=28, labelpad=15)
        ax.set_title(f"CR {target_crs[0]}-{target_crs[1]}  "
                    f"({t_plot_start.strftime('%Y %b %d')} - {t_plot_end.strftime('%Y %b %d')})",
                    fontsize=28, pad=10)
        ax.margins(x=0.005)

    axes[-1].set_xlabel("Date", fontsize=28, labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=26,
              frameon=False, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=[0, 0, 1, 0.95 if n_rows > 1 else 0.86])

    if save_path:
        plt.savefig(save_path, dpi=400)
    return fig


def plot_confusion_matrix(event_df, metrics=("TP", "FP", "FN")):
    """Bar-style confusion-matrix view of the TP/FP/FN counts from
    evaluate_event_detection(). (TN is not defined for this peak-matching
    task -- there's no fixed universe of "non-event" instances to count.)"""
    models = event_df.index.tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(2.5 * len(models) + 2, 5))
    colors = {"TP": "tab:green", "FP": "tab:red", "FN": "tab:orange"}
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, event_df[metric], width, label=metric,
              color=colors.get(metric, "gray"))
        for xi, v in zip(x + (i - 1) * width, event_df[metric]):
            ax.text(xi, v, str(int(v)), ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Event detection: TP / FP / FN per model", fontsize=14)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_peak_bias(all_matched_dict, time_window_hr=24, figsize=(26, 9)):
    """Violin plots of peak timing bias (dt_hr) and speed bias (dv) per model,
    from match_peaks() output. Notebook-only diagnostic, not a published figure."""
    models = list(all_matched_dict.keys())
    dt_data = [np.array([m["dt_hr"] for m in all_matched_dict[k]]) for k in models]
    dv_data = [np.array([m["dv"] for m in all_matched_dict[k]]) for k in models]
    colors = [MODEL_COLORS.get(m, "gray") for m in models]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, wspace=0.25)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rng = np.random.default_rng(42)

    vp1 = ax1.violinplot(dt_data, positions=range(len(models)), showmedians=True,
                         showextrema=False, vert=False)
    for body, c in zip(vp1["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.4)
    vp1["cmedians"].set_color("black"); vp1["cmedians"].set_linewidth(2)
    for i, (d, c) in enumerate(zip(dt_data, colors)):
        ax1.scatter(d, i + rng.uniform(-0.12, 0.12, size=len(d)), color=c, alpha=0.6, s=35, zorder=3)
    ax1.axvline(0, color="gray", lw=1.2, ls="--")
    ax1.set_yticks(range(len(models))); ax1.set_yticklabels(models, fontsize=16)
    ax1.set_xlabel("Timing error  dt [hours]", fontsize=20)
    ax1.set_title(f"Peak Timing Bias (+/-{time_window_hr}h window)", fontsize=22, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=15)
    for i, d in enumerate(dt_data):
        ax1.text(ax1.get_xlim()[1], i, f"n={len(d)}", ha="left", va="center", fontsize=15, color="gray")

    vp2 = ax2.violinplot(dv_data, positions=range(len(models)), showmedians=True, showextrema=False)
    for body, c in zip(vp2["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.4)
    vp2["cmedians"].set_color("black"); vp2["cmedians"].set_linewidth(2)
    for i, (d, c) in enumerate(zip(dv_data, colors)):
        ax2.scatter(i + rng.uniform(-0.12, 0.12, size=len(d)), d, color=c, alpha=0.6, s=35, zorder=3)
    ax2.axhline(0, color="gray", lw=1.2, ls="--")
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(models, rotation=30, ha="right", fontsize=16)
    ax2.set_ylabel("Speed error  dv [km/s]", fontsize=20)
    ax2.set_title(f"Peak Speed Bias (+/-{time_window_hr}h window)", fontsize=22, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=15)
    for i, d in enumerate(dv_data):
        ax2.text(i, ax2.get_ylim()[0], f"n={len(d)}", ha="center", va="top", fontsize=15, color="gray")

    fig.suptitle(f"HSS Peak Bias Analysis  (time window = +/-{time_window_hr} h)",
                fontsize=25, fontweight="bold")
    plt.tight_layout()
    return fig


_FITS_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})\d{2}Z")


def _find_nearest_fits(directory, target_dt, tol_hours=3):
    """Find the 193A FITS file in `directory` whose filename timestamp is
    closest to target_dt (within tol_hours)."""
    tol = timedelta(hours=tol_hours)
    best_f, best_diff = None, timedelta.max
    for f in sorted(glob.glob(os.path.join(directory, "*.fits"))):
        m = _FITS_TS_RE.search(os.path.basename(f))
        if not m:
            continue
        dt = pd.Timestamp(f"{m.group(1)}T{m.group(2)}:{m.group(3)}")
        diff = abs(dt - pd.Timestamp(target_dt))
        if diff <= tol and diff < best_diff:
            best_diff, best_f = diff, f
    if best_f is None:
        raise FileNotFoundError(f"No FITS file within +/-{tol_hours}h of {target_dt} in {directory}")
    return best_f


def plot_speed_profile_cr_with_euv(df, cr_df, panels, series_specs, icme_intervals,
                                   aia_dir="../data/sample", propagation_delay_days=4,
                                   color_prev="mediumpurple", color_now="mediumblue",
                                   save_path=None):
    """
    One row per panel: two AIA 193A EUV images (dt_prev, dt_now -- typically
    27 days apart) with SPoCA CH contours, beside the CR speed profile for
    that panel's Carrington Rotations. Vertical lines mark dt_prev/dt_now on
    the speed panel, with a "+N days" arrow showing the propagation delay
    from dt_now to when its effect should appear in the SR-derived speed.

    panels : list of dicts, each {"label": str, "crs": [cr1, cr2],
             "dt_prev": datetime, "dt_now": datetime}.
    aia_dir : folder containing the 193A FITS files for dt_prev/dt_now of
             every panel (nearest-timestamp match, see _find_nearest_fits).
    """
    from sklearn.metrics import mean_absolute_error  # noqa: F401 (parity with plot_speed_profile_cr imports)

    time_all = df["datetime"]
    icme_mask = make_icme_mask(time_all, icme_intervals)
    has_band = {"max_sqrt_AP", "min_sqrt_AP"}.issubset(df.columns)
    sr_col = next((c for c, l in series_specs if "SR" in l), None)

    sir_dicts = {"OMNI": detect_HSE_blocks(time_all, df["speed"])}
    for colname, label in series_specs:
        sir_dicts[label] = detect_HSE_blocks(time_all, df[colname])

    aia_data = []
    for panel in panels:
        f_prev = _find_nearest_fits(aia_dir, panel["dt_prev"])
        f_now = _find_nearest_fits(aia_dir, panel["dt_now"])
        aia_prev, aia_now = load_and_calibrate(f_prev)[1], load_and_calibrate(f_now)[1]
        poly_prev = get_spoca_ch_union(aia_prev, hours=1)
        poly_now = get_spoca_ch_union(aia_now, hours=1)
        rgba_prev, ext_prev = get_rgba_and_extent(aia_prev, "193")
        rgba_now, ext_now = get_rgba_and_extent(aia_now, "193")
        aia_data.append(dict(aia_prev=aia_prev, rgba_prev=rgba_prev, ext_prev=ext_prev, poly_prev=poly_prev,
                             aia_now=aia_now, rgba_now=rgba_now, ext_now=ext_now, poly_now=poly_now))

    n_rows = len(panels)
    fig = plt.figure(figsize=(34, 5 * n_rows), facecolor="white")
    gs_speed = gridspec.GridSpec(n_rows, 1, left=0.34, right=0.99, hspace=0.3, top=0.93, bottom=0.07)
    gs_euv = gridspec.GridSpec(n_rows, 2, left=0.04, right=0.30, hspace=0.3, wspace=0.05, top=0.93, bottom=0.07)

    axes_speed = []
    for row_idx, (panel, data) in enumerate(zip(panels, aia_data)):
        dt_prev, dt_now = panel["dt_prev"], panel["dt_now"]

        for col, dt, color, key in [(0, dt_prev, color_prev, "prev"), (1, dt_now, color_now, "now")]:
            ax_euv = fig.add_subplot(gs_euv[row_idx, col])
            ax_euv.set_facecolor("black")
            ax_euv.imshow(data[f"rgba_{key}"], origin="lower", extent=data[f"ext_{key}"])
            draw_ch_contour(ax_euv, data[f"aia_{key}"], data[f"poly_{key}"], data[f"ext_{key}"])
            set_hpc_axes(ax_euv, data[f"ext_{key}"], show_xlabel=False, show_ylabel=False,
                        label_fs=1, tick_fs=16, grid=True)
            for spine in ax_euv.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(6)
            ax_euv.set_title(dt.strftime("%Y %b %d %H:%M UT"), fontsize=22, color=color, pad=8)

        ax_speed = fig.add_subplot(gs_speed[row_idx, 0])
        axes_speed.append(ax_speed)

        target_crs = panel["crs"]
        t_starts, t_ends = zip(*[get_cr_date_range(cr_df, cr) for cr in target_crs])
        t_plot_start, t_plot_end = min(t_starts), max(t_ends)
        cr_df_plot = df[(time_all >= t_plot_start) & (time_all < t_plot_end)]
        time_cr = cr_df_plot["datetime"]

        shade_icme(ax_speed, icme_intervals, t0=t_plot_start, t1=t_plot_end, facecolor="gray", alpha=0.4)
        ax_speed.axvline(t_ends[0], color="gray", linestyle=":", linewidth=2, alpha=0.8)
        for cr_num, t_s in zip(target_crs, t_starts):
            ax_speed.text(t_s + timedelta(hours=6), 868, f"CR {cr_num}", ha="left", va="top",
                         fontsize=28, color="gray")

        if has_band and sr_col:
            ax_speed.fill_between(cr_df_plot["datetime"],
                                 cr_df_plot[sr_col] - df.loc[cr_df_plot.index, "min_sqrt_AP"],
                                 cr_df_plot[sr_col] + df.loc[cr_df_plot.index, "max_sqrt_AP"],
                                 color="red", alpha=0.2)

        ax_speed.plot(cr_df_plot["datetime"], cr_df_plot["speed"], color=MODEL_COLORS["OMNI"], lw=3, label="OMNI")
        plot_peaks(ax_speed, sir_dicts["OMNI"], df["speed"], time_all, time_cr, icme_mask,
                  color=MODEL_COLORS["OMNI"], markersize=12)
        for colname, label in series_specs:
            color = MODEL_COLORS.get(label, "gray")
            ls, lw = ("-", 3) if "SR" in label else ("--", 2)
            ax_speed.plot(cr_df_plot["datetime"], cr_df_plot[colname], color=color, lw=lw, linestyle=ls, label=label)
            plot_peaks(ax_speed, sir_dicts[label], df[colname], time_all, time_cr, icme_mask,
                      color=color, markersize=12)

        ax_speed.axvline(dt_prev, color=color_prev, lw=5, ls="--", zorder=10, alpha=0.9)
        ax_speed.axvline(dt_now, color=color_now, lw=5, ls="-", zorder=10, alpha=0.9)

        dt_arrow_end = dt_now + timedelta(days=propagation_delay_days)
        idx = (df["datetime"] - pd.Timestamp(dt_arrow_end)).abs().idxmin()
        arrow_y = float(df.loc[idx, sr_col]) if sr_col and pd.notna(df.loc[idx, sr_col]) else 500.0
        ax_speed.annotate("", xy=(dt_arrow_end, arrow_y), xytext=(dt_now, arrow_y),
                         arrowprops=dict(arrowstyle="-|>", color=color_now, lw=2.5, mutation_scale=18), zorder=14)
        ax_speed.text(dt_now + timedelta(days=propagation_delay_days / 2), arrow_y + 12,
                     f"+{propagation_delay_days} days", ha="center", va="bottom", fontsize=20,
                     color=color_now, fontweight="bold", zorder=15)

        ax_speed.set_xlim(t_plot_start, t_plot_end)
        ax_speed.set_ylim(250, 900)
        ax_speed.set_yticks([400, 600, 800])
        ax_speed.tick_params(axis="y", labelsize=26)
        ax_speed.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax_speed.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_speed.tick_params(axis="x", labelsize=26)
        ax_speed.set_ylabel("Speed [km/s]", fontsize=28, labelpad=15)
        ax_speed.set_title(f"CR {target_crs[0]}-{target_crs[1]}  "
                          f"({t_plot_start.strftime('%Y %b %d')} - {t_plot_end.strftime('%Y %b %d')})",
                          fontsize=28, pad=8)
        ax_speed.margins(x=0.005)
        if row_idx < n_rows - 1:
            ax_speed.set_xlabel("")
        else:
            ax_speed.set_xlabel("Date", fontsize=28, labelpad=10)

    handles, labels = axes_speed[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=26,
              frameon=False, bbox_to_anchor=(0.665, 1.02))

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig
