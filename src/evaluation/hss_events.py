"""
HSS/SIR event detection and evaluation: peak extraction across models,
event-detection metrics (POD/FAR/CSI/BS via time+speed matching windows),
and peak timing/speed bias analysis (Samara-style matching).

Consolidates what used to be reimplemented (and re-reimplemented) across
notebooks/06_HSS_events_verification.ipynb: the detect_HSE_blocks + peak
extraction loop, the TP/FP/FN event matching, and the pairwise peak-bias
matching each had their own copy per cell.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from utils.hse_detection import detect_HSE_blocks, get_peaks
from utils.icme import make_icme_mask
from utils.metrics import event_verification


def add_formula_uncertainty_band(df, a_bands=(30, 60, 90), p_bands=(30, 60, 90), lag="lag4"):
    """
    max_sqrt_AP / min_sqrt_AP = the max/min of sqrt(A_CH_a * P_CH_p) across every
    A/P latitude-band combination at the given lag. Used as a +/- uncertainty
    band around the SR formula's accel term when plotting, showing how much
    the prediction would shift under a different (but still plausible) band
    choice (see notebook 05's latitude-combination analysis).
    """
    df = df.copy()
    values = []
    for a in a_bands:
        for p in p_bands:
            col_A, col_P = f"A_CH{a}_193_{lag}", f"P_CH{p}_211_{lag}"
            if col_A in df.columns and col_P in df.columns:
                values.append(np.sqrt(df[col_A] * df[col_P]))
    combos = pd.concat(values, axis=1)
    df["max_sqrt_AP"] = combos.max(axis=1)
    df["min_sqrt_AP"] = combos.min(axis=1)
    return df


def detect_peaks_for_models(df, model_cols, icme_intervals, t0=None, t1=None,
                            target_col="speed", time_col="datetime"):
    """
    Run detect_HSE_blocks + get_peaks for every column in model_cols (plus the
    observed target_col as "OMNI"), over the window [t0, t1] (defaults to the
    full df range).

    Returns {label: [(timestamp, speed), ...]} keyed by column name
    ("OMNI" for target_col).
    """
    time_all = df[time_col]
    t0 = t0 or time_all.iloc[0]
    t1 = t1 or time_all.iloc[-1]
    time_window = df[(time_all >= t0) & (time_all <= t1)][time_col]

    icme_mask = make_icme_mask(time_all, icme_intervals)

    peaks = {}
    for label, col in {"OMNI": target_col, **model_cols}.items():
        sir_dict = detect_HSE_blocks(time_all, df[col])
        peaks[label] = get_peaks(sir_dict, df[col], time_all, time_window, icme_mask)
    return peaks


def match_peaks_with_window(omni_peaks, model_peaks, time_window=timedelta(hours=24),
                            speed_window=2000):
    """
    Greedy TP matching: a model peak counts as a hit if it falls within
    +/-time_window of an OMNI peak AND within +/-speed_window km/s of that
    peak's speed (each OMNI peak matches at most one model peak).

    Returns (TP, FP, FN) counts.
    """
    TP = 0
    for omni_time, omni_speed in omni_peaks:
        t_start, t_end = omni_time - time_window, omni_time + time_window
        for model_time, model_speed in model_peaks:
            if t_start <= model_time <= t_end and abs(model_speed - omni_speed) <= speed_window:
                TP += 1
                break
    FP = len(model_peaks) - TP
    FN = len(omni_peaks) - TP
    return TP, FP, FN


def evaluate_event_detection(all_peak_lists, time_window=timedelta(hours=24), speed_window=2000):
    """
    TP/FP/FN + POD/FAR/CSI/BS for every model in all_peak_lists (output of
    detect_peaks_for_models) against the "OMNI" entry.
    """
    omni_peaks = all_peak_lists["OMNI"]
    rows = []
    for model, model_peaks in all_peak_lists.items():
        if model == "OMNI":
            continue
        TP, FP, FN = match_peaks_with_window(omni_peaks, model_peaks, time_window, speed_window)
        POD, FAR, CSI, BS = event_verification(TP, FP, FN)
        rows.append({"model": model, "TP": TP, "FP": FP, "FN": FN,
                     "POD": POD, "FAR": FAR, "CSI": CSI, "BS": BS})
    return pd.DataFrame(rows).set_index("model")


def match_peaks(omni_peaks, model_peaks, time_window_hr=24):
    """
    Pairwise nearest-time matching (each model peak used at most once) for
    peak-bias analysis: how early/late and how over/under each matched peak
    is relative to the OMNI peak it's matched to.

    Returns (matched, unmatched_omni) where matched is a list of dicts with
    omni_time/omni_speed/model_time/model_speed/dt_hr/dv.
    """
    window = timedelta(hours=time_window_hr)
    used_model = set()
    matched, unmatched_omni = [], []

    for o_t, o_v in omni_peaks:
        candidates = [
            (abs((m_t - o_t).total_seconds()), i, m_t, m_v)
            for i, (m_t, m_v) in enumerate(model_peaks)
            if abs((m_t - o_t).total_seconds()) <= window.total_seconds() and i not in used_model
        ]
        if candidates:
            candidates.sort()
            _, best_i, m_t, m_v = candidates[0]
            used_model.add(best_i)
            matched.append({
                "omni_time": o_t, "omni_speed": o_v, "model_time": m_t, "model_speed": m_v,
                "dt_hr": (m_t - o_t).total_seconds() / 3600.0, "dv": m_v - o_v,
            })
        else:
            unmatched_omni.append((o_t, o_v))

    return matched, unmatched_omni


def peak_bias_stats(matched, model_name="Model"):
    """Timing (dt_hr) and speed (dv) bias summary stats for a matched-peaks list,
    including Wilcoxon signed-rank test p-values."""
    if not matched:
        return {"model": model_name, "n_hits": 0}

    dt = np.array([m["dt_hr"] for m in matched])
    dv = np.array([m["dv"] for m in matched])

    def signed_pct(arr):
        return np.sum(arr > 0) / len(arr) * 100, np.sum(arr < 0) / len(arr) * 100

    dt_pos_pct, dt_neg_pct = signed_pct(dt)
    dv_pos_pct, dv_neg_pct = signed_pct(dv)
    wt_dt = stats.wilcoxon(dt) if len(dt) >= 10 else (np.nan, np.nan)
    wt_dv = stats.wilcoxon(dv) if len(dv) >= 10 else (np.nan, np.nan)

    return {
        "model": model_name, "n_hits": len(matched),
        "dt_mean_hr": np.mean(dt), "dt_median_hr": np.median(dt), "dt_std_hr": np.std(dt),
        "dt_mae_hr": np.mean(np.abs(dt)), "dt_lead_pct": dt_neg_pct, "dt_lag_pct": dt_pos_pct,
        "dt_wilcoxon_p": wt_dt[1],
        "dv_mean": np.mean(dv), "dv_median": np.median(dv), "dv_std": np.std(dv),
        "dv_mae": np.mean(np.abs(dv)), "dv_over_pct": dv_pos_pct, "dv_under_pct": dv_neg_pct,
        "dv_wilcoxon_p": wt_dv[1],
    }
