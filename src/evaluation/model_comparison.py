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


def _dtw_normalized(y_true, y_pred, window=None):
    """Dynamic Time Warping distance (Samara et al. 2022), z-score normalized."""
    from dtaidistance import dtw as dtw_lib

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 2:
        return np.nan, np.nan

    def zscore(x):
        mu, sigma = np.mean(x), np.std(x)
        return (x - mu) / sigma if sigma > 0 else x - mu

    yt_z = zscore(yt).astype(np.double)
    yp_z = zscore(yp).astype(np.double)

    kwargs = {"window": window} if window is not None else {}
    dist = dtw_lib.distance(yt_z, yp_z, **kwargs)
    return round(dist / n, 6), round(dist, 2)


def _row_metrics(y_true, y_pred, include_dtw=False, dtw_window=None):
    n = len(y_true)
    if n == 0:
        row = {"MAE": np.nan, "RMSE": np.nan, "CC": np.nan, "N": 0}
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        cc = np.corrcoef(y_true, y_pred)[0, 1] if n > 1 else np.nan
        row = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "CC": round(cc, 3), "N": n}
    if include_dtw:
        dtw_norm, dtw_raw = _dtw_normalized(y_true, y_pred, window=dtw_window)
        row["DTW_norm"] = dtw_norm
        row["DTW_raw"] = dtw_raw
    return row


def evaluate_metrics(df, group_by=None, groups=None, target_col="speed",
                     model_cols=None, test_months=(10, 11, 12),
                     include_dtw=False, dtw_window=None):
    """
    Evaluate MAE/RMSE/CC (optionally DTW) on the Oct-Dec test months, ICME
    periods excluded, for each candidate model column.

    group_by : None -> single row per model, over the entire period.
               "year" -> one row per (year, model), 2010-2024.
               "phase" -> one row per (phase, model); `groups` is a dict of
                          {phase_name: [years]} (defaults to SOLAR_CYCLE_PHASES).
    model_cols : defaults to every "best*" column plus speed_p27/wsa_enlil/eswf2
                 that are present in df.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[df["datetime"].dt.month.isin(test_months)]
    if "is_ICME" in df.columns:
        df = df[~df["is_ICME"]]

    if model_cols is None:
        model_cols = [c for c in df.columns if c.startswith("best")]
        model_cols += [c for c in ["speed_p27", "wsa_enlil", "eswf2"] if c in df.columns]

    if group_by == "year":
        years = sorted(df["datetime"].dt.year.unique())
        group_iter = [(y, df[df["datetime"].dt.year == y]) for y in years]
        group_col = "year"
    elif group_by == "phase":
        groups = groups or SOLAR_CYCLE_PHASES
        group_iter = [(name, df[df["datetime"].dt.year.isin(yrs)]) for name, yrs in groups.items()]
        group_col = "phase"
    else:
        group_iter = [(None, df)]
        group_col = None

    rows = []
    for group_val, sub_df in group_iter:
        sub_df = sub_df[[target_col] + model_cols].dropna()
        for model in model_cols:
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
