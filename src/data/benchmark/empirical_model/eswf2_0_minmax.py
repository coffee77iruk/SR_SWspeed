import sys, os
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import re

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

def eswf2_minmax(df, cr_df, column_name, new_col_name="eswf2"):
    df = df.copy()
    cr_df = cr_df.copy()

    df['datetime'] = pd.to_datetime(df['datetime'])
    cr_df['Start Date'] = pd.to_datetime(cr_df['Start Date'])
    cr_df['End Date'] = pd.to_datetime(cr_df['End Date'])

    df[new_col_name] = np.nan
    df["A_min"] = np.nan
    df["A_max"] = np.nan
    df["v_min"] = np.nan
    df["v_max"] = np.nan

    df_start = df["datetime"].min()
    df_end   = df["datetime"].max()

    cr_use = cr_df[(cr_df["End Date"] >= df_start) & (cr_df["Start Date"] <= df_end)].reset_index(drop=True)
    if len(cr_use) < 4:
        return df[['datetime', 'speed', column_name, new_col_name, 'A_min', 'A_max', 'v_min', 'v_max']].copy()
    
    # ---- Loop: previous 3 CR -> coefficients -> next CR ----
    for i in range(3, len(cr_use)):
        prev_start = cr_use.loc[i-3, "Start Date"]
        prev_end   = cr_use.loc[i-1, "End Date"]

        prev_mask = (df["datetime"] >= prev_start) & (df["datetime"] <= prev_end)
        prev_data = df.loc[prev_mask, ["speed", column_name]].dropna()

        if len(prev_data) < 10:
            continue

        # 1. Reject top/bottom 5%
        """
        First, 5% of the highest/lowest fractional coronal hole areas A(t + τ) 
        and measured solar wind speeds are rejected.

        """
        speed_low, speed_high = prev_data["speed"].quantile([0.05, 0.95])
        A_low, A_high = prev_data[column_name].quantile([0.05, 0.95])

        filtered = prev_data[
            (prev_data["speed"] >= speed_low) & (prev_data["speed"] <= speed_high) &
            (prev_data[column_name] >= A_low) & (prev_data[column_name] <= A_high)
        ]

        if len(filtered) == 0:
            continue

        # 2. Define coefficients
        """
        Second, from the remaining 90% the minimum/maximum values, denoted by Amax, Amin and vmax, vmin, 
        are computed and used as new model coefficients for the following CR.

        """
        v_min = filtered["speed"].min()
        v_max = filtered["speed"].max()
        A_min = filtered[column_name].min()
        A_max = filtered[column_name].max()

        if A_max == A_min:
            continue

        curr_start = cr_use.loc[i, "Start Date"]
        curr_end   = cr_use.loc[i, "End Date"]

        curr_mask = (df["datetime"] >= curr_start) & (df["datetime"] <= curr_end)

        df.loc[curr_mask, new_col_name] = (
            v_min +
            (v_max - v_min) / (A_max - A_min) *
            (df.loc[curr_mask, column_name] - A_min)
        )

        df.loc[curr_mask, "A_min"] = A_min
        df.loc[curr_mask, "A_max"] = A_max
        df.loc[curr_mask, "v_min"] = v_min
        df.loc[curr_mask, "v_max"] = v_max

    return df[['datetime', 'speed', column_name, new_col_name, 'A_min', 'A_max', 'v_min', 'v_max']].copy()