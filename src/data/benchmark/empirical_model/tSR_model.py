import sys, os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from datetime import datetime, timedelta
import re
import warnings

sys.path.append(os.path.abspath("../../"))
from data.benchmark.wsa_enlil.cr_data import fetch_cr_table

URL = "https://space.umd.edu/pm/crn/"
cr_df = fetch_cr_table(URL)
cr_df.head()

def preprocess_ch_df(file: str):
    df = pd.read_csv(file)

    target_chan = "211"
    drop_cols = [c for c in df.columns if re.match(rf"^A_CH\d+_{target_chan}$", c)]
    df = df.drop(columns=drop_cols)
    cols = [c for c in df.columns if c.startswith("P_CH")]

    # Sigma-clipping for P_CH parameters
    for _ in range(4):
        for col in cols:
            mu, sigma = df[col].mean(), df[col].std()
            mask = (df[col] < mu - 3*sigma) | (df[col] > mu + 3*sigma) | (df[col] <= 0)
            df.loc[mask, col] = np.nan

    # 21:00 UT corresponds to the daily dark calibration time for AIA.
    mask_21 = df['datetime'].str.contains("T21:", na=False)
    ch_param_cols = df.columns.difference(["datetime"])
    df.loc[mask_21, ch_param_cols] = np.nan
    return df.iloc[:-1] # Remove last row (2025-01-01 00:00:00)

def load_omni_data(path: str):
    row = []
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            year, doy, hour, speed = line.split()
            year, doy, hour, speed = int(year), int(doy), int(hour), float(speed)

            if year >= 2010 and hour % 1 == 0:
                dt = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour)
                row.append({'datetime': dt, 'speed': speed})
    return pd.DataFrame(row)

def build_sr_df(omni_df, ch_df):
    omni, ch = omni_df.copy(), ch_df.copy()
    omni['datetime'] = pd.to_datetime(omni['datetime'])
    ch['datetime']   = pd.to_datetime(ch['datetime'])

    df = (
        pd.merge(omni, ch, on="datetime", how="inner")
          .sort_values("datetime")
          .reset_index(drop=True)
    )
    df.loc[df['speed'] >= 1000, 'speed'] = np.nan

    # Merge with CH parameters
    shifts = {'lag4': 96}
    ch_param_cols = df.columns.difference(["datetime", "speed"])
    for param in ch_param_cols:
        for name, shift in shifts.items():
            col_name = f"{param}_{name}"
            df[col_name] = df[param].shift(shift)


    return df.reset_index(drop=True)

def tSR_model(df, cr_df, 
              col_A="A_CH60_193_lag4",
              col_P="P_CH30_211_lag4",
              col_v_prev="speed_p27",
              new_col_name="best_tSR"):
    """
    Tunable SR model: v = a1 * sqrt(A * P) + a2 * sqrt(v_prev)

    Parameters
    ----------
    df        : DataFrame with 'datetime', 'speed', col_A, col_P, col_v_prev
    cr_df     : Carrington rotation table (Start Date / End Date)
    col_A     : coronal hole area column  (default: A_CH60_193_lag4)
    col_P     : CH parameter column       (default: P_CH30_211_lag4)
    col_v_prev: persistence speed column  (default: speed_p27)
    new_col_name : output column name
    """

    df = df.copy()
    cr_df = cr_df.copy()

    df['datetime'] = pd.to_datetime(df['datetime'])
    cr_df['Start Date'] = pd.to_datetime(cr_df['Start Date'])
    cr_df['End Date'] = pd.to_datetime(cr_df['End Date'])

    df[new_col_name] = np.nan
    df["a1"] = np.nan
    df["a2"] = np.nan

    df_start = df["datetime"].min()
    df_end   = df["datetime"].max()

    cr_use = (
        cr_df[(cr_df["End Date"] >= df_start) & (cr_df["Start Date"] <= df_end)]
        .reset_index(drop=True)
    )
    if len(cr_use) < 4:
        return df[["datetime", "speed", new_col_name, "a1", "a2"]].copy()

    # ── model formula ──────────────────────────────────────────────────────────
    def _tsr(X, a1, a2):
        A, P, v_prev = X
        return a1 * np.sqrt(np.maximum(A * P, 0)) + a2 * np.sqrt(np.maximum(v_prev, 0))

    # ── loop: previous 3 CR → fit → predict next CR ───────────────────────────
    for i in range(3, len(cr_use)):

        # ── 1. collect previous 3-CR training window ──────────────────────────
        prev_start = cr_use.loc[i - 3, "Start Date"]
        prev_end   = cr_use.loc[i - 1, "End Date"]

        prev_mask  = (df["datetime"] >= prev_start) & (df["datetime"] <= prev_end)
        prev_data  = df.loc[prev_mask, ["speed", col_A, col_P, col_v_prev]].dropna()

        if len(prev_data) < 10:
            continue

        # ── 2. reject top/bottom 5% of speed (same as minmax_model) ──────────
        speed_low, speed_high = prev_data["speed"].quantile([0.05, 0.95])
        filtered = prev_data[
            prev_data["speed"].between(speed_low, speed_high)
        ]
        if len(filtered) < 10:
            continue

        X_train = (
            filtered[col_A].values,
            filtered[col_P].values,
            filtered[col_v_prev].values,
        )
        y_train = filtered["speed"].values

        # ── 3. curve_fit for (a1, a2) ─────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _tsr,
                    X_train,
                    y_train,
                    p0=[1.0, 19.2873],            # initial guess from original SR
                    bounds=([0.0, 0.0],
                            [10.0, 35.0]),    # physically meaningful bounds
                    maxfev=20000,
                )
            a1_fit, a2_fit = popt
        except (RuntimeError, ValueError):
            continue

        # ── 4. predict current CR ─────────────────────────────────────────────
        curr_start = cr_use.loc[i, "Start Date"]
        curr_end   = cr_use.loc[i, "End Date"]
        curr_mask  = (df["datetime"] >= curr_start) & (df["datetime"] <= curr_end)

        A_c      = df.loc[curr_mask, col_A]
        P_c      = df.loc[curr_mask, col_P]
        v_prev_c = df.loc[curr_mask, col_v_prev]

        valid = A_c.notna() & P_c.notna() & v_prev_c.notna()

        df.loc[curr_mask & valid, new_col_name] = _tsr(
            (A_c[valid].values, P_c[valid].values, v_prev_c[valid].values),
            a1_fit, a2_fit,
        )

        # store fitted coefficients for diagnostics
        df.loc[curr_mask, "a1"] = a1_fit
        df.loc[curr_mask, "a2"] = a2_fit

    return df[["datetime", "speed", new_col_name, "a1", "a2"]].copy()