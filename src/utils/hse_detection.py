"""
High-Speed Enhancement (HSE) / Stream Interaction Region (SIR) detection
and associated plotting utilities.

Functions
---------
detect_HSE_blocks : Identify SIR events in a solar wind speed time series.
plot_peaks        : Mark peak speeds of detected SIRs on a matplotlib Axes.
shade_icme        : Shade ICME intervals on a matplotlib Axes.
get_cr_date_range : Return the start/end date of a Carrington Rotation.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


def detect_HSE_blocks(Time: pd.Series, Speed: pd.Series) -> dict:
    """
    Detect SIR events following the algorithm of Jian et al. (2006).

    Steps
    -----
    1. Candidate points: speed > speed[i-24h] + 50 km/s.
    2. Drop isolated single-point candidates; group contiguous blocks as HSEs.
    3-5. For each HSE find Vmin (2 days before start) and Vmax (1 day after end).
    6. Merge SIRs separated by < 18 h.
    9. Reject: Vmin >= 500, Vmax <= 400, or Vmax - Vmin < 100 km/s.

    Returns a dict with keys 'dates', 'speeds', 'index', each a list of
    [SIR_start, SIR_end] pairs.
    """
    candidates = [
        i for i in range(24, len(Time))
        if Speed.iloc[i] > Speed.iloc[i - 24] + 50
        and not Time.iloc[i - 24:i + 1].isna().any()
    ]

    groups = []
    if candidates:
        current = [candidates[0]]
        for idx in candidates[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                if len(current) > 1:
                    groups.append(current)
                current = [idx]
        if len(current) > 1:
            groups.append(current)

    event_groups = {'dates': [], 'speeds': [], 'index': []}
    for grp in groups:
        event_groups['dates'].append([Time.iloc[i] for i in grp])
        event_groups['speeds'].append([Speed.iloc[i] for i in grp])
        event_groups['index'].append(grp)

    SIR_dict = {'dates': [], 'speeds': [], 'index': []}
    for i_list in event_groups['index']:
        start_idx, end_idx = i_list[0], i_list[-1]
        start_idx = max(start_idx, 48)

        seg_min      = Speed.iloc[start_idx - 48:start_idx + 1]
        argmin_index = np.argmin(seg_min)
        min_index    = start_idx - 48 + argmin_index
        min_date     = Time.iloc[min_index]
        min_speed    = seg_min.iloc[argmin_index]

        seg_max      = Speed.iloc[end_idx:end_idx + 25]
        argmax_index = np.argmax(seg_max)
        max_index    = end_idx + argmax_index
        max_date     = Time.iloc[max_index]
        max_speed    = seg_max.iloc[argmax_index]

        SIR_dict['dates'].append([min_date, max_date])
        SIR_dict['speeds'].append([min_speed, max_speed])
        SIR_dict['index'].append([min_index, max_index])

    events   = list(zip(SIR_dict['dates'], SIR_dict['speeds'], SIR_dict['index']))
    merged   = []
    dt_merge = timedelta(hours=18)

    for dates, speed, idx in events:
        if merged and dates[0] - merged[-1]['dates'][1] <= dt_merge:
            merged[-1]['dates'][1]  = dates[1]
            merged[-1]['speeds'][1] = speed[1]
            merged[-1]['index'][1]  = idx[1]
        else:
            merged.append({
                'dates':  [dates[0], dates[1]],
                'speeds': [speed[0], speed[1]],
                'index':  [idx[0],   idx[1]],
            })

    merged2 = {'dates': [], 'speeds': [], 'index': []}
    for m in merged:
        idx0, idx1 = m['index']
        if idx1 - idx0 > 2:
            merged2['dates'].append(m['dates'])
            merged2['speeds'].append(m['speeds'])
            merged2['index'].append(m['index'])

    result = {'dates': [], 'speeds': [], 'index': []}
    for i in range(len(merged2['index'])):
        SIR_start, SIR_end = merged2['index'][i]
        SIR_speeds = Speed.iloc[SIR_start:SIR_end + 1]
        v_min, v_max = np.min(SIR_speeds), np.max(SIR_speeds)
        if v_min >= 500 or v_max <= 400 or (v_max - v_min) < 100:
            continue
        result['dates'].append(merged2['dates'][i])
        result['speeds'].append(merged2['speeds'][i])
        result['index'].append(merged2['index'][i])

    return result


def plot_peaks(ax, sir_dict: dict, speed_series: pd.Series,
               time_all: pd.Series, time_year: pd.Series,
               icme_mask: np.ndarray, color: str,
               markersize: int = 15) -> list:
    """
    Plot the peak speed marker (▼) for each SIR event that falls within
    time_year and is not masked as an ICME.

    Returns a list of (timestamp, speed) tuples for each detected peak.
    """
    peak_list = []
    t0, t1    = time_year.iloc[0], time_year.iloc[-1]

    for i in range(len(sir_dict['index'])):
        SIR_start, SIR_end = sir_dict['index'][i]
        SIR_speeds = speed_series.iloc[SIR_start:SIR_end + 1]

        if np.all(np.isnan(SIR_speeds.values)):
            continue

        max_index = SIR_start + np.nanargmax(SIR_speeds.values)
        if icme_mask[max_index]:
            continue

        v_max = float(speed_series.iloc[max_index])
        t_max = time_all.iloc[max_index]

        if not (t0 <= t_max <= t1):
            continue

        ax.plot(t_max, v_max, color=color, marker='v',
                markersize=markersize, linestyle='None')
        peak_list.append((t_max, v_max))

    return peak_list


def shade_icme(ax, icme_intervals: list, t0, t1,
               facecolor: str = 'gray', alpha: float = 0.4) -> None:
    """Shade each ICME interval that overlaps [t0, t1] on the given Axes."""
    for icme_start, icme_end in icme_intervals:
        if icme_end < t0 or icme_start > t1:
            continue
        ax.axvspan(max(icme_start, t0), min(icme_end, t1),
                   facecolor=facecolor, alpha=alpha, linewidth=0)


def get_cr_date_range(cr_df: pd.DataFrame, cr_number: int):
    """
    Return (start_date, end_date) for the given Carrington Rotation number.

    cr_df must contain columns 'Carrington Rotation Number', 'Start Date',
    and 'End Date'.
    """
    cr_df = cr_df.copy()
    cr_df['Start Date'] = pd.to_datetime(cr_df['Start Date'])
    cr_df['End Date']   = pd.to_datetime(cr_df['End Date'])

    row = cr_df[cr_df['Carrington Rotation Number'] == cr_number]
    if row.empty:
        raise ValueError(f"CR {cr_number} not found in cr_df.")

    return row['Start Date'].iloc[0], row['End Date'].iloc[0]
