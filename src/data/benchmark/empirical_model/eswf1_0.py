import numpy as np
import pandas as pd


def eswf1(df, column_name, new_col_name="eswf1", c0=350.0,
                 train_months=range(1, 10), quantile_low=0.05, quantile_high=0.95):
    """
    ESWF 1.0: Amplitude-optimized method with fixed c0, c1.
    Rotter et al. (2012), Vršnak et al. (2007).

    c0 = 350 km/s (slow solar wind background, physically known constant)
    c1 = (v_max - c0) / A_max  (derived from entire training set, fixed)

    Parameters
    ----------
    df            : DataFrame with 'datetime', 'speed', column_name
    column_name   : lagged CH area (예: "A_CH90_193_lag4")
    new_col_name  : predicted 
    c0            : slow wind background speed (default: 350 km/s)
    train_months  : month used in training (default: 1–9 month)
    quantile_low  : remove lower quantile (default: 0.05)
    quantile_high : remove upper quantile (default: 0.95)

    Returns
    -------
    DataFrame with columns:
        ['datetime', 'speed', column_name, new_col_name, 'A_max', 'v_max', 'c0', 'c1']
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    df[new_col_name] = np.nan
    df["A_max"]      = np.nan
    df["v_max"]      = np.nan
    df["c0"]         = np.nan
    df["c1"]         = np.nan

    # ── Step 1. Training set (Jan. – Sep. for each years) ──────────────────────────────
    train_mask = df['datetime'].dt.month.isin(train_months)
    train_data = df.loc[train_mask, ["speed", column_name]].dropna()
    train_data = train_data[train_data[column_name] > 0]

    if len(train_data) < 10:
        return df[['datetime', 'speed', column_name,
                   new_col_name, 'A_max', 'v_max', 'c0', 'c1']].copy()

    # ── Step 2. Reject top/bottom 5% ───────────────────────────────────────────
    speed_low,  speed_high = train_data["speed"].quantile([quantile_low, quantile_high])
    A_low,      A_high     = train_data[column_name].quantile([quantile_low, quantile_high])

    filtered = train_data[
        (train_data["speed"]     >= speed_low)  & (train_data["speed"]     <= speed_high) &
        (train_data[column_name] >= A_low)       & (train_data[column_name] <= A_high)
    ]

    if len(filtered) == 0:
        return df[['datetime', 'speed', column_name,
                   new_col_name, 'A_max', 'v_max', 'c0', 'c1']].copy()

    # ── Step 3. Calculate fixed coefficient ──────────────────────────────────────────────
    # c0 = 350 km/s (slow wind background, physically known constant)
    # c1 = (v_max - c0) / A_max
    v_max = filtered["speed"].max()
    A_max = filtered[column_name].max()

    c1 = (v_max - c0) / A_max

    print(f"[ESWF 1.0] c0 = {c0:.1f} km/s (fixed)")
    print(f"[ESWF 1.0] c1 = {c1:.1f} km/s  (v_max={v_max:.1f}, A_max={A_max:.6f})")

    # ── Step 4. prediction ──────────────────────────────────────────────
    valid_mask = df[column_name].notna() & (df[column_name] > 0)

    df.loc[valid_mask, new_col_name] = c0 + c1 * df.loc[valid_mask, column_name]
    df.loc[valid_mask, "A_max"]      = A_max
    df.loc[valid_mask, "v_max"]      = v_max
    df.loc[valid_mask, "c0"]         = c0
    df.loc[valid_mask, "c1"]         = c1

    return df[['datetime', 'speed', column_name,
               new_col_name, 'A_max', 'v_max', 'c0', 'c1']].copy()
