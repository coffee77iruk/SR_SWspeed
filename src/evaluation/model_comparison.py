"""
Build the SR-vs-baseline comparison dataset and evaluate performance against it.

Consolidates what used to be reimplemented separately across
notebooks/05_verifing_performance.ipynb: data loading (now reuses
utils.ch_processing / utils.icme instead of duplicating them), attaching the
three baseline models (ESWF, WSA-ENLIL, 27-day persistence) plus the
SR-derived formula, and a single parameterized metrics evaluator (entire /
yearly / phase, optionally with DTW) instead of three near-identical copies.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.ch_processing import preprocess_ch_df, load_omni_data, build_sr_df
from utils.icme import fetch_icme_events, mask_icme_events
from data.benchmark.wsa_enlil.cr_data import fetch_cr_table
from data.benchmark.empirical_model.eswf2_0_minmax import eswf2_minmax
from data.benchmark.wsa_enlil.wsa_enlil_ccmc import WSA_ENLIL

CR_URL = "https://space.umd.edu/pm/crn/"

PERSISTENCE_SHIFTS = {"speed_p3": 72, "speed_p4": 96, "speed_p5": 120, "speed_p27": 648}

# Published formula: v_t = sqrt(A_CH60,t-4d * P_CH30,t-4d) + sqrt(v_t-27d * 372.1)
SR_FORMULA = "sqrt(A_CH60_193_lag4 * P_CH30_211_lag4) + sqrt(speed_p27 * 372.1075472)"

SOLAR_CYCLE_PHASES = {
    "Rising":    list(range(2010, 2013)) + list(range(2021, 2024)),
    "Maximum":   [2013, 2014, 2024],
    "Declining": list(range(2015, 2019)),
    "Minimum":   [2019, 2020],
}


def build_comparison_df(data_dir="../data"):
    """
    Build the ICME-masked comparison dataset: CH/OMNI features + lag/persistence
    columns + eswf2 + wsa_enlil + best_sr (the published formula), ready for
    evaluate_metrics() / compute_binned_stats().

    Returns (df, cr_df).
    """
    ch_df = preprocess_ch_df(os.path.join(data_dir, "interim", "CH_param_cad1.csv"))
    omni_df = load_omni_data(os.path.join(data_dir, "processed", "omni2_2000-2024.lst"))
    df = build_sr_df(omni_df, ch_df, persistence_shifts=PERSISTENCE_SHIFTS)

    cr_df = fetch_cr_table(CR_URL)

    lag4_cols = [c for c in df.columns if c.endswith("_lag4")]
    base = df[["datetime", "speed"] + lag4_cols].copy()
    df["eswf2"] = eswf2_minmax(base, cr_df, column_name="A_CH90_193_lag4")["eswf2"]

    wsa_df = WSA_ENLIL(shift_days=4).sort_values("datetime").drop_duplicates("datetime", keep="last")
    df = df.merge(
        wsa_df[["datetime", "speed"]].rename(columns={"speed": "wsa_enlil"}),
        on="datetime", how="left",
    )

    icme_events = fetch_icme_events()
    df = mask_icme_events(df, icme_events, persistence_shifts=PERSISTENCE_SHIFTS)

    df["best_sr"] = df.eval(SR_FORMULA, engine="python", local_dict={"np": np})

    return df, cr_df


def _dtw_score(y_true, y_pred, window=None):
    """
    Dynamic Time Warping distance (Samara et al. 2022; Edward-Inatimi et al. 2026),
    computed directly on the physical-unit (km/s) speed series -- no z-scoring.
    window is a Sakoe-Chiba band size in samples (e.g. 48 for +-2 days of hourly
    data, the window Samara et al. 2022 found appropriate for solar wind HSS
    timing uncertainty).

    This is a relative metric intended for comparing models evaluated on the
    same series (same N, e.g. different model columns within one Table-2 phase
    row) -- like Samara's and Edward-Inatimi's own DTW score, it is not
    length-normalized, so raw values should not be compared across groups with
    different N (e.g. across phases with different row counts).
    """
    from dtaidistance import dtw as dtw_lib

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask].astype(np.double), y_pred[mask].astype(np.double)
    if len(yt) < 2:
        return np.nan

    kwargs = {"window": window} if window is not None else {}
    return round(dtw_lib.distance(yt, yp, **kwargs), 2)


def _row_metrics(y_true, y_pred, include_dtw=False, dtw_window=None):
    n = len(y_true)
    if n == 0:
        row = {"MAE": np.nan, "RMSE": np.nan, "CC": np.nan, "N": 0}
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # A constant y_pred (e.g. the "av" baseline) has zero true variance,
        # but np.corrcoef's two-pass variance sum can land a few ULPs off
        # exact zero for large n, silently producing a meaningless ~1e-15
        # "correlation" instead of the mathematically correct undefined/NaN.
        # np.ptp involves no summation, so it stays exactly 0 for a constant.
        constant_pred = n > 1 and np.ptp(y_pred) == 0
        if n > 1 and not constant_pred:
            cc = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            cc = np.nan
        row = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "CC": round(cc, 3), "N": n}
    if include_dtw:
        row["DTW"] = _dtw_score(y_true, y_pred, window=dtw_window)
    return row


def evaluate_metrics(df, group_by=None, groups=None, target_col="speed",
                     model_cols=None, test_months=(10, 11, 12),
                     include_dtw=False, dtw_window=None, train_months=range(1, 10)):
    """
    Evaluate MAE/RMSE/CC (optionally DTW) on the Oct-Dec test months, ICME
    periods excluded, for each candidate model column.

    group_by : None -> single row per model, over the entire period.
               "year" -> one row per (year, model), 2010-2024.
               "phase" -> one row per (phase, model); `groups` is a dict of
                          {phase_name: [years]} (defaults to SOLAR_CYCLE_PHASES).
    model_cols : defaults to every "best*" column plus speed_p27/wsa_enlil/eswf2/av
                 that are present in df (av is always available -- see below).

    "av" (average-prediction baseline, Collin et al. 2025): a constant equal
    to the mean OMNI speed over the Jan-Sep training months, ICME periods
    excluded -- the simplest possible "no information" baseline, included to
    show that a good RMSE alone doesn't imply real skill (CC for a constant
    is undefined/near-zero by construction). Not a real df column, since its
    value depends on which years the current group spans: the single
    2010-2024 training mean for the entire-period row (group_by=None), that
    year's own training mean for group_by="year", or that phase's own
    training mean for group_by="phase" -- each row uses only the training
    data from its own group's years, so a per-phase row doesn't leak
    information a single global constant wouldn't have.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    train_df = df[df["datetime"].dt.month.isin(train_months)]
    if "is_ICME" in train_df.columns:
        train_df = train_df[~train_df["is_ICME"]]

    df = df[df["datetime"].dt.month.isin(test_months)]
    if "is_ICME" in df.columns:
        df = df[~df["is_ICME"]]

    if model_cols is None:
        model_cols = [c for c in df.columns if c.startswith("best")]
        model_cols += [c for c in ["speed_p27", "wsa_enlil", "eswf2"] if c in df.columns]
        model_cols += ["av"]

    use_av = "av" in model_cols
    real_model_cols = [c for c in model_cols if c != "av"]

    if group_by == "year":
        years = sorted(df["datetime"].dt.year.unique())
        group_iter = [(y, df[df["datetime"].dt.year == y], [y]) for y in years]
        group_col = "year"
    elif group_by == "phase":
        groups = groups or SOLAR_CYCLE_PHASES
        group_iter = [(name, df[df["datetime"].dt.year.isin(yrs)], yrs) for name, yrs in groups.items()]
        group_col = "phase"
    else:
        group_iter = [(None, df, sorted(df["datetime"].dt.year.unique()))]
        group_col = None

    rows = []
    for group_val, sub_df, group_years in group_iter:
        sub_df = sub_df.copy()
        cols = list(real_model_cols)
        if use_av:
            av_const = train_df.loc[train_df["datetime"].dt.year.isin(group_years), target_col].mean()
            sub_df["av"] = av_const
            cols = cols + ["av"]

        sub_df = sub_df[[target_col] + cols].dropna()
        for model in cols:
            with np.errstate(invalid="ignore"):
                row = _row_metrics(
                    sub_df[target_col].values, sub_df[model].values,
                    include_dtw=include_dtw, dtw_window=dtw_window,
                )
            row["model"] = model
            if group_col:
                row[group_col] = group_val
            rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in [group_col, "model"] if c]
    return out.sort_values(sort_cols).reset_index(drop=True)


def evaluate_latitude_combinations(df, a_bands=(30, 60, 90), p_bands=(30, 60, 90),
                                   bg_const=372.1075472, groups=None):
    """
    Build the SR formula for every A_CH/P_CH latitude-band combination
    (sr_A{a}_P{p} = sqrt(A_CH{a}_193_lag4 * P_CH{p}_211_lag4) + sqrt(speed_p27 * bg_const))
    and evaluate each one per solar-cycle phase (plus "Entire" = all years),
    reusing evaluate_metrics rather than a separate copy of the metrics logic.

    Returns a DataFrame like evaluate_metrics(group_by="phase") with extra
    A_band/P_band integer columns, so the published A60xP30 formula's rank
    among all combinations can be checked per phase.
    """
    df = df.copy()
    combo_cols = []
    for a in a_bands:
        for p in p_bands:
            col = f"sr_A{a}_P{p}"
            expr = f"sqrt(A_CH{a}_193_lag4 * P_CH{p}_211_lag4) + sqrt(speed_p27 * {bg_const})"
            df[col] = df.eval(expr, engine="python", local_dict={"np": np})
            combo_cols.append(col)

    groups = dict(groups or SOLAR_CYCLE_PHASES)
    groups["Entire"] = sorted({y for years in groups.values() for y in years})

    out = evaluate_metrics(df, group_by="phase", groups=groups, model_cols=combo_cols)
    out["A_band"] = out["model"].str.extract(r"sr_A(\d+)_P\d+").astype(int)
    out["P_band"] = out["model"].str.extract(r"sr_A\d+_P(\d+)").astype(int)
    return out


def compute_binned_stats(df, model_specs, speed_bins, min_samples=10,
                         n_bootstrap=1000, ci=95, seed=42):
    """
    Bin observations by OMNI speed and compute MAE/RMSE/Bias/CC per bin per
    model, with bootstrap confidence intervals.

    model_specs : list of (column_name, display_label) tuples.
    Returns (stats_dict, bin_labels) where stats_dict[label] is a DataFrame
    indexed by bin, with columns MAE/RMSE/Bias/CC/N and _lo/_hi CI widths for
    MAE/RMSE/Bias.
    """
    from scipy.stats import pearsonr

    data = df[~df["is_ICME"]].copy() if "is_ICME" in df.columns else df.copy()
    data = data.dropna(subset=["speed"])

    bin_labels = [f"{speed_bins[i]}-{speed_bins[i + 1]}" for i in range(len(speed_bins) - 1)]
    data["speed_bin"] = pd.cut(data["speed"], bins=speed_bins, labels=bin_labels, right=False)

    rng = np.random.default_rng(seed)
    alpha = (100 - ci) / 2

    stats_dict = {}
    for colname, label in model_specs:
        rows = []
        sub = data.dropna(subset=[colname])

        for bl in bin_labels:
            mask = sub["speed_bin"] == bl
            obs = sub.loc[mask, "speed"].values
            pred = sub.loc[mask, colname].values
            n = len(obs)

            if n < min_samples:
                rows.append({"bin": bl, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan,
                            "CC": np.nan, "N": n, "MAE_lo": np.nan, "MAE_hi": np.nan,
                            "RMSE_lo": np.nan, "RMSE_hi": np.nan, "Bias_lo": np.nan, "Bias_hi": np.nan})
                continue

            mae = np.mean(np.abs(pred - obs))
            rmse = np.sqrt(np.mean((pred - obs) ** 2))
            bias = np.mean(pred - obs)
            cc = pearsonr(obs, pred)[0] if n >= 3 else np.nan

            idx = np.arange(n)
            boot_mae, boot_rmse, boot_bias = [], [], []
            for _ in range(n_bootstrap):
                s = rng.choice(idx, size=n, replace=True)
                o_b, p_b = obs[s], pred[s]
                boot_mae.append(np.mean(np.abs(p_b - o_b)))
                boot_rmse.append(np.sqrt(np.mean((p_b - o_b) ** 2)))
                boot_bias.append(np.mean(p_b - o_b))

            rows.append({
                "bin": bl, "MAE": mae, "RMSE": rmse, "Bias": bias, "CC": cc, "N": n,
                "MAE_lo": mae - np.percentile(boot_mae, alpha),
                "MAE_hi": np.percentile(boot_mae, 100 - alpha) - mae,
                "RMSE_lo": rmse - np.percentile(boot_rmse, alpha),
                "RMSE_hi": np.percentile(boot_rmse, 100 - alpha) - rmse,
                "Bias_lo": bias - np.percentile(boot_bias, alpha),
                "Bias_hi": np.percentile(boot_bias, 100 - alpha) - bias,
            })

        stats_dict[label] = pd.DataFrame(rows).set_index("bin")

    return stats_dict, bin_labels


def load_sunspot_numbers(path_m, path_ms, start_year=2010, end_year=2024):
    """
    Parse SIDC monthly (path_m) and 13-month-smoothed (path_ms) sunspot number
    files (SN_m_tot_V2.0.txt / SN_ms_tot_V2.0.txt format).

    Returns (date, sunspot_num_m, sunspot_num_ms).
    """
    date, sunspot_num_m, sunspot_num_ms = [], [], []

    with open(path_m, "r") as f:
        for line in f:
            try:
                dt = datetime.strptime(line[:7], "%Y %m")
                if start_year <= dt.year <= end_year:
                    date.append(dt)
                    sunspot_num_m.append(float(line[18:23]))
            except ValueError:
                continue

    with open(path_ms, "r") as f:
        for line in f:
            try:
                dt = datetime.strptime(line[:7], "%Y %m")
                if start_year <= dt.year <= end_year:
                    sunspot_num_ms.append(float(line[18:23]))
            except ValueError:
                continue

    return date, sunspot_num_m, sunspot_num_ms
