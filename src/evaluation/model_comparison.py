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

# AIA excludes every 21:00 UT frame daily (dark-frame calibration), which
# lands as a structural, guaranteed 1-hour gap (20:00 -> 22:00) in the
# best_sr timeline too (the 96h/4-day CH lag preserves hour-of-day). Confirmed
# empirically: of 1380 gaps in the entire-period test-month, ICME-excluded
# dataset, 1079 (78%) are exactly this single missing hour, 1051 of those
# ending at hour 22. DTW_GAP_TOLERANCE_HOURS controls how many consecutive
# missing hours evaluate_metrics() will bridge (linear interpolation) rather
# than treat as a DTW block break -- see _build_dtw_blocks().
DTW_GAP_TOLERANCE_HOURS = 1
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
    Dynamic Time Warping distance, computed directly on the physical-unit
    (km/s) speed series -- no z-scoring. Uses dtaidistance's inner_dist=
    "euclidean" mode, i.e. D(i,j) = |s_i - q_j| + min(D(i-1,j-1), D(i-1,j),
    D(i,j-1)) with the raw cumulative sum returned as-is (no final sqrt).
    This matches both the equation in Samara et al. (2022) and the actual
    implementation in their cited reference code
    (github.com/SamaraEvangelia/DTW_ForSolarWindEvaluation) -- dtaidistance's
    *default* inner distance is squared Euclidean with a single sqrt at the
    end instead, which is a materially different number, not this one.

    window is a Sakoe-Chiba band size in samples (e.g. 48 for +-2 days of
    hourly data).

    Returns (distance, path_length) for the block passed in. distance is the
    raw cumulative sum (not divided by anything). path_length is the number
    of matched pairs in the optimal warping path -- >= max(len(yt), len(yp)),
    and strictly larger whenever a singularity (a point matched to more than
    one point in the other series) occurs. evaluate_metrics() sums both
    across every calendar-contiguous block in a group and divides
    distance-sum by path-length-sum, so the reported average is a true "cost
    per matched pair" (sum of terms / count of terms) rather than diluting
    singularity-heavy alignments across a plain point count that doesn't
    reflect how many times each point was actually matched.
    """
    from dtaidistance import dtw as dtw_lib

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask].astype(np.double), y_pred[mask].astype(np.double)
    if len(yt) < 2:
        return np.nan, 0

    kwargs = {"window": window} if window is not None else {}
    path, dist = dtw_lib.warping_path(yt, yp, include_distance=True, inner_dist="euclidean", **kwargs)
    return round(dist, 2), len(path)


def _build_dtw_blocks(sub_df, target_col, value_cols, icme_intervals,
                      gap_tolerance_hours=DTW_GAP_TOLERANCE_HOURS):
    """
    Partition a chronologically sorted, already-dropna'd sub_df into
    DTW-ready blocks, adding a 'block_id' column.

    A gap in the nominal 1-hour cadence is bridged (linearly interpolated,
    the (a+b)/2 average for a single missing hour) rather than starting a
    new block, PROVIDED it is at most gap_tolerance_hours consecutive
    missing hours AND doesn't overlap any ICME interval. This absorbs the
    routine daily 21:00 UT AIA dark-frame exclusion (and other short,
    incidental single/few-hour data gaps) so they don't fragment a real
    contiguous observing period into artificial per-day blocks.

    Any gap that overlaps an ICME-masked interval always starts a new block
    and is never bridged, regardless of its duration -- ICME periods are a
    deliberate exclusion (isolating ambient/CH-driven solar wind from
    CME-driven perturbations), so interpolating a smooth line across one
    would fabricate physically wrong values, not fill in noise.

    icme_intervals : list of (start, end) Timestamp tuples, e.g. from
                     utils.icme.fetch_icme_events().
    """
    sub_df = sub_df.reset_index(drop=True)
    dt = sub_df["datetime"]

    icme_starts = np.array([s.value for s, _ in icme_intervals])
    icme_ends = np.array([e.value for _, e in icme_intervals])

    def _overlaps_icme(t_start, t_end):
        if len(icme_starts) == 0:
            return False
        return bool(np.any((icme_starts <= t_end.value) & (icme_ends >= t_start.value)))

    block_ids = [0]
    bridge_rows = []
    current_block = 0
    for i in range(1, len(dt)):
        prev_t, curr_t = dt.iloc[i - 1], dt.iloc[i]
        gap_hours = (curr_t - prev_t).total_seconds() / 3600.0
        is_hourly_grid = abs(gap_hours - round(gap_hours)) < 1e-6
        missing_hours = int(round(gap_hours)) - 1 if is_hourly_grid else None

        if gap_hours <= 1.0:
            pass  # nominal 1-hour cadence, same block
        elif (
            is_hourly_grid
            and missing_hours <= gap_tolerance_hours
            and not _overlaps_icme(prev_t, curr_t)
        ):
            prev_row, curr_row = sub_df.iloc[i - 1], sub_df.iloc[i]
            missing_times = pd.date_range(prev_t, curr_t, freq="1h")[1:-1]
            for j, t in enumerate(missing_times, start=1):
                frac = j / (missing_hours + 1)
                new_row = {"datetime": t, "block_id": current_block}
                for c in [target_col] + list(value_cols):
                    new_row[c] = prev_row[c] + frac * (curr_row[c] - prev_row[c])
                bridge_rows.append(new_row)
        else:
            current_block += 1
        block_ids.append(current_block)

    out = sub_df.copy()
    out["block_id"] = block_ids
    if bridge_rows:
        out = pd.concat([out, pd.DataFrame(bridge_rows)], ignore_index=True)
    return out.sort_values(["block_id", "datetime"]).reset_index(drop=True)


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
        row["DTW_mean"] = _dtw_score(y_true, y_pred, window=dtw_window)
    return row


def evaluate_metrics(df, group_by=None, groups=None, target_col="speed",
                     model_cols=None, test_months=(10, 11, 12),
                     include_dtw=False, dtw_window=None, train_months=range(1, 10),
                     dtw_gap_tolerance_hours=DTW_GAP_TOLERANCE_HOURS):
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

    dtw_gap_tolerance_hours : when include_dtw, gaps of at most this many
    consecutive missing hours are bridged (linearly interpolated) rather
    than starting a new DTW block -- see _build_dtw_blocks(). Gaps that
    overlap an ICME interval are always hard block breaks regardless of
    this setting.
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

    icme_intervals = fetch_icme_events() if include_dtw else []

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

        dtw_cols = cols
        if include_dtw:
            # Reference constant for SSF_mean: the Oct-Dec test period's own
            # true mean speed (scoped to this group's years), captured before
            # any column subsetting/dropna below. Deliberately different from
            # "av" (which uses the Jan-Sep training mean to avoid leaking
            # test-period information) -- SSF_mean asks "how much better than
            # trivially guessing the true test-period average", which is
            # meant to use in-sample information, the same way R^2's null
            # model uses the test set's own mean.
            sub_df["_test_mean"] = sub_df[target_col].mean()
            dtw_cols = cols + ["_test_mean"]

        sub_df = sub_df[["datetime", target_col] + dtw_cols].dropna().sort_values("datetime")

        if include_dtw:
            # DTW must only ever treat genuinely time-adjacent samples as
            # adjacent. Both the Jan-Sep gap between years and any
            # ICME-masked or otherwise-missing hours within a single Oct-Dec
            # season leave array positions sitting next to each other despite
            # being far apart in real time. Most within-season gaps, though,
            # are just the routine daily 21:00 UT AIA dark-frame exclusion
            # (a single missing hour, propagated to best_sr via its 96h/4-day
            # lag) -- fragmenting a block over that isn't meaningful, so those
            # are bridged (linearly interpolated) instead of split. See
            # _build_dtw_blocks() for the exact rule, including the hard
            # ICME-never-bridged guarantee.
            sub_df_dtw = _build_dtw_blocks(
                sub_df, target_col, dtw_cols, icme_intervals,
                gap_tolerance_hours=dtw_gap_tolerance_hours,
            )
            # Raw block-summed DTW cost (not yet normalized), one sum per
            # column -- reused both for DTW_mean (divided by that column's
            # own path length, an absolute km/s score) and for the SSF
            # ratios (Samara et al. (2022)'s ratio-normalization: a model's
            # raw DTW cost divided by a reference model's own raw DTW cost
            # against the same OMNI series over the same blocks).
            raw_sum, path_sum = {}, {}
            for c in dtw_cols:
                d_tot, p_tot = 0.0, 0
                for _, block in sub_df_dtw.groupby("block_id"):
                    if len(block) < 2:
                        continue
                    d, p = _dtw_score(block[target_col].values, block[c].values, window=dtw_window)
                    if not np.isnan(d):
                        d_tot += d
                        p_tot += p
                raw_sum[c] = d_tot
                path_sum[c] = p_tot

        for model in cols:
            with np.errstate(invalid="ignore"):
                row = _row_metrics(
                    sub_df[target_col].values, sub_df[model].values,
                    include_dtw=False,
                )
            if include_dtw:
                # Report cost-per-matched-pair (total cost / total warping
                # path length), not the raw sum: a sum has no natural
                # comparison point across groups with different N (e.g.
                # across phases), while this average stays in km/s and is
                # directly comparable to MAE/RMSE in the same table -- both
                # within a row and across rows. Dividing by path length
                # rather than plain point count N correctly dilutes
                # singularities (points matched more than once) across the
                # extra matched pairs they actually produced, instead of
                # concentrating that cost onto a point count that doesn't
                # reflect the repeated matches.
                row["DTW_mean"] = round(raw_sum[model] / path_sum[model], 2) if path_sum[model] > 0 else np.nan
                # SSF (Sequence Similarity Factor, Samara et al. 2022): a
                # ratio, so it's computed from the raw block-summed costs
                # directly (both numerator and denominator use the same
                # blocks/window), not from the already path-length-divided
                # DTW_mean values.
                ref_mean_sum = raw_sum.get("_test_mean", 0)
                row["SSF_mean"] = round(raw_sum[model] / ref_mean_sum, 3) if ref_mean_sum > 0 else np.nan
                ref_27d_sum = raw_sum.get("speed_p27", 0)
                row["SSF_27days"] = round(raw_sum[model] / ref_27d_sum, 3) if ref_27d_sum > 0 else np.nan
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
