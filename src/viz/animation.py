"""
Frame-by-frame animation (PNG sequence -> MP4/GIF) pairing an AIA 193A CH
image (current + 27-days-earlier side by side) with a synchronized SR-derived
speed profile panel. Presentation material, not a published figure.

Requires ffmpeg on PATH for frames_to_video()/frames_to_gif().
"""

import os
import re
import glob
import shutil
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from utils.hse_detection import detect_HSE_blocks, plot_peaks, shade_icme, get_cr_date_range
from utils.icme import make_icme_mask
from viz.ch_parameters import load_and_calibrate, get_spoca_ch_union, get_rgba_and_extent, set_hpc_axes, draw_ch_contour
from viz.hss_events import MODEL_COLORS

_FITS_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})\d{2}Z")


def parse_datetime_from_fits(path, round_to_hour=True):
    """Parse the observation datetime from an AIA FITS filename, optionally
    rounded to the nearest hour (>=30 min rounds up)."""
    m = _FITS_TS_RE.search(os.path.basename(path))
    if not m:
        return None
    dt = datetime.strptime(f"{m.group(1)}T{m.group(2)}{m.group(3)}", "%Y-%m-%dT%H%M")
    if round_to_hour:
        dt = dt.replace(minute=0, second=0)
        if int(m.group(3)) >= 30:
            dt += timedelta(hours=1)
    return dt


def collect_files(directory_template, date_start, date_end, interval_hours=6):
    """
    Build a regular time grid from date_start to date_end (step=interval_hours)
    and, for each grid point, find the nearest FITS file (within
    interval_hours/2) under directory_template (a path with a 4-digit year
    that gets substituted per year spanned).

    Returns a list of (grid_datetime, filepath) tuples.
    """
    all_files = []
    for year in range(date_start.year, date_end.year + 1):
        year_dir = re.sub(r"\d{4}(?=[^0-9]*$)", str(year), directory_template)
        all_files.extend(glob.glob(os.path.join(year_dir, "*.fits")))

    dt_to_file = {}
    for f in sorted(all_files):
        dt = parse_datetime_from_fits(f)
        if dt is not None:
            dt_to_file[dt] = f
    if not dt_to_file:
        return []

    file_times = sorted(dt_to_file.keys())
    tol = timedelta(hours=interval_hours / 2)
    grid_start = date_start.replace(
        hour=(date_start.hour // interval_hours) * interval_hours, minute=0, second=0, microsecond=0)

    result = []
    grid_dt = grid_start
    while grid_dt <= date_end:
        if grid_dt >= date_start:
            candidates = [(abs((ft - grid_dt).total_seconds()), ft) for ft in file_times
                         if abs((ft - grid_dt).total_seconds()) <= tol.total_seconds()]
            if candidates:
                _, best = min(candidates)
                result.append((grid_dt, dt_to_file[best]))
        grid_dt += timedelta(hours=interval_hours)
    return result


def match_files_27day(files_now, files_prev):
    """Pair up (dt, path) lists index-by-index: [(dt_now, f_now, dt_prev, f_prev), ...]."""
    n = min(len(files_now), len(files_prev))
    return [(files_now[i][0], files_now[i][1], files_prev[i][0], files_prev[i][1]) for i in range(n)]


def build_cr_animation_figure(df, cr_df, cr_pair, icme_intervals, series_specs,
                              sr_col="best_sr", propagation_delay_days=4):
    """
    Build the static (non-animated) parts of one animation frame's figure:
    a speed-profile panel on top (all series, ICME shading, HSS peaks, SR
    uncertainty band) and two empty AIA image axes below (prev / now), plus
    the artists that get updated per-frame.

    Returns (fig, ax_speed, ax_prev, ax_now, state) where state is a dict of
    mutable per-frame artist handles used by update_cr_animation_frame().
    """
    time_all = df["datetime"]
    icme_mask = make_icme_mask(time_all, icme_intervals)

    t_starts, t_ends = zip(*[get_cr_date_range(cr_df, cr) for cr in cr_pair])
    t_plot_start, t_plot_end = min(t_starts), max(t_ends)
    cr_df_plot = df[(time_all >= t_plot_start) & (time_all < t_plot_end)]

    fig = plt.figure(figsize=(30, 20), facecolor="none")
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 10], hspace=0.18, wspace=0.04,
                           top=0.90, bottom=0.07, left=0.07, right=0.99)
    ax_speed = fig.add_subplot(gs[0, :])
    ax_prev = fig.add_subplot(gs[1, 0])
    ax_now = fig.add_subplot(gs[1, 1])

    ax_speed.set_facecolor("white")
    shade_icme(ax_speed, icme_intervals, t_plot_start, t_plot_end)
    ax_speed.axvline(t_ends[0], color="gray", linestyle=":", linewidth=2, alpha=0.8)
    for cr_num, t_s in zip(cr_pair, t_starts):
        ax_speed.text(t_s + timedelta(hours=6), 868, f"CR {cr_num}", ha="left", va="top",
                     fontsize=30, color="gray")

    for colname, label in series_specs:
        color = MODEL_COLORS.get(label, "gray")
        ls, lw = ("-", 3) if "SR" in label else ("--", 2)
        alpha = 0.25 if colname == sr_col else 1.0
        ax_speed.plot(cr_df_plot["datetime"], cr_df_plot[colname], color=color, lw=lw,
                     linestyle=ls, label=label, alpha=alpha)
        sir_dict = detect_HSE_blocks(time_all, df[colname])
        plot_peaks(ax_speed, sir_dict, df[colname], time_all, cr_df_plot["datetime"], icme_mask,
                  color=color, markersize=12)

    sr_left_line, = ax_speed.plot([], [], color="red", lw=3, linestyle="-", zorder=8)

    ax_speed.set_xlim(t_plot_start, t_plot_end)
    ax_speed.set_ylim(250, 900)
    ax_speed.set_yticks([400, 600, 800])
    ax_speed.tick_params(axis="y", labelsize=28)
    ax_speed.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_speed.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_speed.tick_params(axis="x", labelsize=28)
    ax_speed.set_ylabel("Speed [km/s]", fontsize=30, labelpad=12)
    ax_speed.set_xlabel("Date", fontsize=30, labelpad=8)
    ax_speed.margins(x=0.005)

    handles, labels = ax_speed.get_legend_handles_labels()
    leg = ax_speed.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.4),
                          ncol=len(labels), fontsize=30, frameon=False)
    for handle, label in zip(leg.legend_handles, labels):
        if "SR" in label:
            handle.set_alpha(1.0)

    ax_speed.set_title(
        f"CR {cr_pair[0]}-{cr_pair[1]}  ({t_plot_start.strftime('%Y %b %d')} - {t_plot_end.strftime('%Y %b %d')})",
        fontsize=30, pad=10)

    for ax in (ax_prev, ax_now):
        ax.set_facecolor("black")

    state = dict(cr_df_plot=cr_df_plot, sr_left_line=sr_left_line, dot=None, band=None,
                arrow=None, text=None, sr_col=sr_col, propagation_delay=timedelta(days=propagation_delay_days))
    return fig, ax_speed, ax_prev, ax_now, state


def update_cr_animation_frame(df, fig, ax_speed, ax_prev, ax_now, state, dt_now, f_now, dt_prev, f_prev):
    """Update one frame in-place: reload the two AIA panels for (dt_prev, dt_now)
    and advance the speed panel's "revealed so far" line/band/arrow."""
    wave_now, aia_now = load_and_calibrate(f_now)
    wave_prev, aia_prev = load_and_calibrate(f_prev)
    poly_now = get_spoca_ch_union(aia_now, hours=1)
    poly_prev = get_spoca_ch_union(aia_prev, hours=1)
    rgba_now, ext_now = get_rgba_and_extent(aia_now, wave_now)
    rgba_prev, ext_prev = get_rgba_and_extent(aia_prev, wave_prev)

    for ax in (ax_prev, ax_now):
        ax.cla()
        ax.set_facecolor("black")

    ax_prev.imshow(rgba_prev, origin="lower", extent=ext_prev)
    ax_now.imshow(rgba_now, origin="lower", extent=ext_now)
    draw_ch_contour(ax_prev, aia_prev, poly_prev, ext_prev)
    draw_ch_contour(ax_now, aia_now, poly_now, ext_now)

    set_hpc_axes(ax_prev, ext_prev, show_xlabel=True, show_ylabel=True, label_fs=30, tick_fs=28)
    set_hpc_axes(ax_now, ext_now, show_xlabel=True, show_ylabel=False, label_fs=30, tick_fs=28)
    for spine in ax_now.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(7)

    ax_prev.set_title(f"{dt_prev.strftime('%Y %b %d %H:%M UT')}  (27 days before)", fontsize=30, pad=12)
    ax_now.set_title(dt_now.strftime("%Y %b %d %H:%M UT"), fontsize=30, color="blue", pad=12)

    for key in ("dot", "arrow", "text"):
        if state[key] is not None:
            state[key].remove()
            state[key] = None

    t_arr = pd.Timestamp(dt_now) + state["propagation_delay"]
    cr_df_plot = state["cr_df_plot"]
    idx_arr = (df["datetime"] - t_arr).abs().idxmin()
    sr_val = df.loc[idx_arr, state["sr_col"]]

    if pd.notna(sr_val):
        state["dot"], = ax_speed.plot(t_arr, sr_val, marker="o", color="red", markersize=18, zorder=15,
                                      linestyle="None", markeredgecolor="darkred", markeredgewidth=1.5)
        state["arrow"] = ax_speed.annotate(
            "", xy=(t_arr, sr_val), xytext=(dt_now, sr_val),
            arrowprops=dict(arrowstyle="-|>", color="blue", lw=3.0, mutation_scale=18), zorder=14)
        days = state["propagation_delay"].days
        state["text"] = ax_speed.text(t_arr - state["propagation_delay"] / 2, sr_val + 18, f"+{days} days",
                                      ha="center", va="bottom", fontsize=22, color="blue",
                                      fontweight="bold", zorder=16)

    mask_left = cr_df_plot["datetime"] <= t_arr
    left = cr_df_plot.loc[mask_left]
    state["sr_left_line"].set_data(left["datetime"], left[state["sr_col"]])

    if state["band"] is not None:
        state["band"].remove()
        state["band"] = None
    if not left.empty and {"max_sqrt_AP", "min_sqrt_AP"}.issubset(df.columns):
        state["band"] = ax_speed.fill_between(
            left["datetime"], left[state["sr_col"]] - df.loc[left.index, "min_sqrt_AP"],
            left[state["sr_col"]] + df.loc[left.index, "max_sqrt_AP"], color="red", alpha=0.2, zorder=5)


def save_cr_animation_frames(df, cr_df, cr_pair, icme_intervals, series_specs, matched_files,
                             output_dir, sr_col="best_sr", propagation_delay_days=4, dpi=150):
    """Render+save one PNG per (dt_now, f_now, dt_prev, f_prev) in matched_files
    (see collect_files()/match_files_27day()). Returns the number of frames saved."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax_speed, ax_prev, ax_now, state = build_cr_animation_figure(
        df, cr_df, cr_pair, icme_intervals, series_specs, sr_col, propagation_delay_days)

    saved = 0
    for i, (dt_now, f_now, dt_prev, f_prev) in enumerate(matched_files):
        try:
            update_cr_animation_frame(df, fig, ax_speed, ax_prev, ax_now, state, dt_now, f_now, dt_prev, f_prev)
            fig.savefig(os.path.join(output_dir, f"frame_{i:04d}.png"), dpi=dpi,
                       facecolor="none", bbox_inches="tight")
            saved += 1
        except Exception as e:
            print(f"  [{i + 1}/{len(matched_files)}] skipped -- {e}")

    plt.close(fig)
    return saved


def _write_concat_list(frame_dir, fps):
    """Write an ffmpeg concat-demuxer list with explicit per-frame durations,
    covering whichever frame_*.png files actually exist (save_cr_animation_frames
    skips frames it fails to render, so numbering can have gaps)."""
    png_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not png_files:
        raise RuntimeError(f"no frame_*.png files found in {frame_dir}")
    concat_path = os.path.join(frame_dir, "_concat_list.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for p in png_files:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")
            f.write(f"duration {1 / fps:.6f}\n")
        f.write(f"file '{png_files[-1].replace(chr(92), '/')}'\n")
    return concat_path


def frames_to_video(frame_dir, output_path, fps=6):
    """ffmpeg: frame_*.png sequence -> H.264 MP4 via the concat demuxer.
    Requires ffmpeg on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    concat_path = _write_concat_list(frame_dir, fps)
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
          "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_path)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return output_path


def frames_to_gif(frame_dir, output_path, fps=6, scale=0.8, width=4500):
    """ffmpeg: frame_*.png sequence -> high-quality palette-based GIF via the
    concat demuxer (two-pass: palettegen, then paletteuse). Requires ffmpeg
    on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    concat_path = _write_concat_list(frame_dir, fps)
    palette_path = os.path.join(frame_dir, "_palette.png")
    w = int(width * scale)
    w = w if w % 2 == 0 else w - 1

    r1 = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                         "-vf", f"scale={w}:-1:flags=lanczos,palettegen=stats_mode=full", palette_path],
                        capture_output=True, text=True)
    if r1.returncode != 0:
        os.remove(concat_path)
        raise RuntimeError(f"ffmpeg palette generation failed:\n{r1.stderr}")

    r2 = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path, "-i", palette_path,
                         "-lavfi", f"scale={w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                         output_path], capture_output=True, text=True)
    os.remove(concat_path)
    if os.path.exists(palette_path):
        os.remove(palette_path)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg GIF encoding failed:\n{r2.stderr}")
    return output_path
