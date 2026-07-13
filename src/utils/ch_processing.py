"""
Solar wind / coronal hole data loading and preprocessing utilities.

Functions
---------
preprocess_ch_df   : Load and clean a CH parameter CSV file.
load_omni_data     : Parse the OMNI hourly solar wind speed text file.
build_sr_df        : Merge OMNI and CH data, add lag / persistence features.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def preprocess_ch_df(file: str) -> pd.DataFrame:
    """
    Load a CH parameter CSV and apply sigma-clipping to P_CH columns.

    - Drops A_CH*_211 area columns (kept only for 193).
    - Applies 4-round 3-sigma clipping to all P_CH* brightness columns.
    - Masks the 21:00 UT row (AIA daily dark calibration artefact).
    - Drops the last row (2025-01-01 00:00:00 boundary).
    """
    df = pd.read_csv(file)

    target_chan = "211"
    drop_cols = [c for c in df.columns if re.match(rf"^A_CH\d+_{target_chan}$", c)]
    df = df.drop(columns=drop_cols)
    cols = [c for c in df.columns if c.startswith("P_CH")]

    for _ in range(4):
        for col in cols:
            mu, sigma = df[col].mean(), df[col].std()
            mask = (df[col] < mu - 3 * sigma) | (df[col] > mu + 3 * sigma) | (df[col] <= 0)
            df.loc[mask, col] = np.nan

    # 21:00 UT = AIA daily dark calibration time
    mask_21 = df['datetime'].str.contains("T21:", na=False)
    ch_param_cols = df.columns.difference(["datetime"])
    df.loc[mask_21, ch_param_cols] = np.nan

    return df.iloc[:-1]  # remove last row (2025-01-01 00:00:00)


def load_omni_data(path: str) -> pd.DataFrame:
    """
    Parse the OMNI hourly solar wind speed text file (omni2_*.lst format).

    Returns a DataFrame with columns ['datetime', 'speed'].
    """
    rows = []
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            year, doy, hour, speed = line.split()
            year, doy, hour, speed = int(year), int(doy), int(hour), float(speed)
            if year >= 2010:
                dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)
                rows.append({'datetime': dt, 'speed': speed})
    return pd.DataFrame(rows)


def build_sr_df(omni_df: pd.DataFrame,
                ch_df: pd.DataFrame,
                persistence_shifts: dict) -> pd.DataFrame:
    """
    Merge OMNI and CH DataFrames, then add lagged CH features and
    persistence speed columns.

    Parameters
    ----------
    omni_df            : DataFrame with ['datetime', 'speed'].
    ch_df              : DataFrame with ['datetime', ...CH params...].
    persistence_shifts : dict mapping column name → shift in hours,
                         e.g. {'speed_p27': 648}.

    Returns
    -------
    Merged DataFrame sorted by datetime with lag and persistence columns added.
    """
    omni = omni_df.copy()
    ch   = ch_df.copy()
    omni['datetime'] = pd.to_datetime(omni['datetime'])
    ch['datetime']   = pd.to_datetime(ch['datetime'])

    df = (
        pd.merge(omni, ch, on="datetime", how="inner")
          .sort_values("datetime")
          .reset_index(drop=True)
    )
    df.loc[df['speed'] >= 1000, 'speed'] = np.nan

    shifts = {'lag3': 72, 'lag3p5': 84, 'lag4': 96, 'lag4p5': 108, 'lag5': 120}
    ch_param_cols = df.columns.difference(["datetime", "speed"])
    for param in ch_param_cols:
        for name, shift in shifts.items():
            df[f"{param}_{name}"] = df[param].shift(shift)

    for name, shift in persistence_shifts.items():
        df[name] = df['speed'].shift(shift)

    return df.reset_index(drop=True)
