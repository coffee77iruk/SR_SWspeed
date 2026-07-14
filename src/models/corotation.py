"""
Corotation-based projection for the SR-derived formula's inputs (Extension 2:
1 AU longitude-wide prediction).

The base Earth formula uses A_CH60/P_CH30 lagged by 4 days and OMNI speed
lagged by 27 days (~1 Carrington rotation, recurrence assumption). To predict
at a different 1 AU longitude (e.g. STEREO-A), the same source region's most
recent Earth observation may be more than one rotation stale, so the "fresh"
4-day/27-day lag isn't available - only the "one rotation earlier" value is
(e.g. lag31 = lag4 + 27d). This module fits/applies a linear correction

    target ~ alpha + beta * stale

instead of assuming perfect persistence (alpha=0, beta=1), to project a stale
observation forward by one Carrington rotation.

Step A (this module + notebooks/supplementary/corotation_1au/01_regression_self_consistency.ipynb)
only validated this on Earth's own data (no Carrington-longitude/time mapping).
Regression correction turned out worse than naive persistence (regression-to-
-the-mean attenuation).

Step B (the longitude-mapping functions below +
notebooks/supplementary/corotation_1au/02_longitude_reconstruction_map.ipynb) builds an
actual Carrington-longitude x time reconstruction: Earth's own historical
hourly series, reindexed by which Carrington longitude Earth was facing at
each observation time, so "value at longitude L, time t" can be looked up as
"the value Earth observed the last time it faced longitude L at/near t" -
plain nearest-occurrence persistence, no regression (consistent with Step A's
finding that a linear correction hurts more than it helps).
"""

import numpy as np
import pandas as pd


def fit_linear_regression(x, y):
    """
    Fit target ~ alpha + beta * stale via least squares, dropping rows where
    either side is NaN. Returns (alpha, beta, r_squared).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid, y_valid = x[valid], y[valid]

    beta, alpha = np.polyfit(x_valid, y_valid, deg=1)
    y_hat = alpha + beta * x_valid
    ss_res = np.sum((y_valid - y_hat) ** 2)
    ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return alpha, beta, r_squared


def project_forward(stale_values, alpha: float, beta: float) -> np.ndarray:
    """Apply a fitted alpha/beta correction to project a stale (1-rotation-old)
    observation forward to the target (fresh) lag."""
    return alpha + beta * np.asarray(stale_values, dtype=float)


# ---------------------------------------------------------------------------
# Step B: Carrington-longitude <-> time mapping and reconstruction
# ---------------------------------------------------------------------------

def carrington_longitude_at_time(times, cr_df, cr_start_col: str = "Start Date",
                                  cr_end_col: str = "End Date") -> np.ndarray:
    """
    Earth's sub-solar Carrington longitude at each timestamp in `times`.

    Longitude decreases linearly from 360 to 0 across each Carrington
    rotation (`cr_df` row): frac = (t - start) / (end - start),
    longitude = (360 * (1 - frac)) % 360.

    Parameters
    ----------
    times : array-like of datetime-like
    cr_df : DataFrame with columns [cr_start_col, cr_end_col] (e.g. fetch_cr_table()
        output), need not be pre-sorted.

    Returns
    -------
    np.ndarray of float longitudes (degrees, 0-360), same length/order as `times`.
    Timestamps before the first or after the last CR in `cr_df` are clipped to
    the nearest edge CR (extrapolated).
    """
    times = pd.to_datetime(pd.Index(times)).to_numpy()
    cr = cr_df.sort_values(cr_start_col).reset_index(drop=True)
    starts = cr[cr_start_col].to_numpy()
    ends = cr[cr_end_col].to_numpy()

    idx = np.searchsorted(starts, times, side="right") - 1
    idx = np.clip(idx, 0, len(cr) - 1)

    start_sel = starts[idx]
    end_sel = ends[idx]
    frac = (times - start_sel) / (end_sel - start_sel)
    longitude = (360.0 * (1.0 - frac.astype(float))) % 360.0
    return longitude


def nearest_time_for_longitude(target_lon: float, reference_time, cr_df,
                                cr_start_col: str = "Start Date",
                                cr_end_col: str = "End Date") -> pd.Timestamp:
    """
    Find the timestamp closest to `reference_time` (any rotation, past or
    future) at which Earth faced Carrington longitude `target_lon`.

    Within any CR row [start, end), longitude hits `target_lon` exactly once,
    at t = start + (1 - target_lon/360) * (end - start). Candidates are
    computed for the CR containing `reference_time` and its immediate
    neighbours (+/-1 rotation), and the closest one is returned - this
    correctly picks the previous-rotation occurrence when `target_lon` is
    "behind" reference_time's own longitude within its CR, and the
    next/previous-CR occurrence otherwise.
    """
    reference_time = pd.Timestamp(reference_time)
    cr = cr_df.sort_values(cr_start_col).reset_index(drop=True)
    starts = cr[cr_start_col].to_numpy()
    ends = cr[cr_end_col].to_numpy()

    idx0 = int(np.clip(np.searchsorted(starts, np.datetime64(reference_time), side="right") - 1,
                        0, len(cr) - 1))
    frac = 1.0 - (target_lon % 360.0) / 360.0

    candidates = []
    for di in (-1, 0, 1):
        i = idx0 + di
        if i < 0 or i >= len(cr):
            continue
        start, end = pd.Timestamp(starts[i]), pd.Timestamp(ends[i])
        candidates.append(start + frac * (end - start))

    return min(candidates, key=lambda t: abs((t - reference_time).total_seconds()))


def reconstruct_longitude_map(source_df: pd.DataFrame, value_col: str, cr_df: pd.DataFrame,
                               time_grid, lon_grid, time_col: str = "datetime",
                               cr_start_col: str = "Start Date", cr_end_col: str = "End Date") -> np.ndarray:
    """
    Build a (time x longitude) reconstruction grid of `value_col` from Earth's
    own historical hourly series `source_df`, using nearest-occurrence
    corotation persistence (no regression - see module docstring).

    For each (reference time, target longitude) cell, finds the nearest
    historical time Earth faced that longitude (via the same +/-1 rotation
    candidate search as `nearest_time_for_longitude`, vectorized over
    `time_grid`) and looks up `value_col` at that time (nearest-hour index
    into `source_df`, which must be a continuous hourly series sorted by
    `time_col`).

    Parameters
    ----------
    source_df : hourly, sorted, continuous DataFrame with [time_col, value_col]
    time_grid : array-like of reference times (output rows)
    lon_grid : array-like of longitudes in degrees, 0-360 (output columns)

    Returns
    -------
    np.ndarray of shape (len(time_grid), len(lon_grid))
    """
    cr = cr_df.sort_values(cr_start_col).reset_index(drop=True)
    cr_starts = cr[cr_start_col].to_numpy()
    cr_ends = cr[cr_end_col].to_numpy()
    n_cr = len(cr)

    time_grid = pd.to_datetime(pd.Index(time_grid)).to_numpy()
    lon_grid = np.asarray(lon_grid, dtype=float)

    src_times = pd.to_datetime(source_df[time_col]).to_numpy()
    src_values = source_df[value_col].to_numpy(dtype=float)
    t0 = src_times[0]
    n_src = len(src_values)

    def lookup_nearest(t_query):
        hours = (t_query - t0) / np.timedelta64(1, "h")
        idx = np.clip(np.round(hours).astype(int), 0, n_src - 1)
        return src_values[idx]

    ref_idx = np.clip(np.searchsorted(cr_starts, time_grid, side="right") - 1, 0, n_cr - 1)

    out = np.full((len(time_grid), len(lon_grid)), np.nan)
    for j, lon in enumerate(lon_grid):
        frac = 1.0 - (lon % 360.0) / 360.0
        best_val = np.full(len(time_grid), np.nan)
        best_dt = np.full(len(time_grid), np.inf)
        for di in (-1, 0, 1):
            i = np.clip(ref_idx + di, 0, n_cr - 1)
            start, end = cr_starts[i], cr_ends[i]
            t_cand = start + frac * (end - start)
            dt = np.abs((t_cand - time_grid) / np.timedelta64(1, "h"))
            val = lookup_nearest(t_cand)
            better = dt < best_dt
            best_dt = np.where(better, dt, best_dt)
            best_val = np.where(better, val, best_val)
        out[:, j] = best_val
    return out
