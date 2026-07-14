"""
Generalized SR-derived solar wind speed formula.

v_t = a * (A_CH * P_CH)^alpha + b * (v_persist27 * v0)^beta

Reduces to the published fixed formula
sqrt(A_CH60_193_lag4 * P_CH30_211_lag4) + sqrt(speed_p27 * 372.1075472)
when a=b=1, alpha=beta=0.5, v0=372.1075472.
"""

import os
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "a": 1.0,
    "b": 1.0,
    "alpha": 0.5,
    "beta": 0.5,
    "v0": 372.1075472,
}

# Same channel / lag combination as the published fixed formula.
FEATURE_COLS = {
    "a_ch": "A_CH60_193_lag4",
    "p_ch": "P_CH30_211_lag4",
    "persist": "speed_p27",
}

_DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "SR_training_entire.csv")
)


def generalized_formula(a_ch, p_ch, persist, a=1.0, b=1.0, alpha=0.5, beta=0.5, v0=372.1075472):
    """Vectorized evaluation of the generalized SR formula."""
    a_ch, p_ch, persist = np.asarray(a_ch, dtype=float), np.asarray(p_ch, dtype=float), np.asarray(persist, dtype=float)
    return a * (a_ch * p_ch) ** alpha + b * (persist * v0) ** beta


def _load_split(month_start: int, month_end: int, csv_path: str, ch_threshold: float) -> pd.DataFrame:
    needed = ["datetime", "speed"] + list(FEATURE_COLS.values())
    df = pd.read_csv(csv_path, usecols=needed)
    df["datetime"] = pd.to_datetime(df["datetime"])

    split_df = df[df["datetime"].dt.month.between(month_start, month_end)].copy()
    split_df = split_df.dropna(subset=list(FEATURE_COLS.values())).reset_index(drop=True)

    split_df["is_ch_present"] = split_df[FEATURE_COLS["a_ch"]] >= ch_threshold
    return split_df


def load_train_set(csv_path: str = _DEFAULT_DATA_PATH, ch_threshold: float = 0.01) -> pd.DataFrame:
    """
    Load the Jan-Sep training set with the three feature columns required by
    generalized_formula(), dropping rows with NaN in any of them (ICME
    periods and CH-processing gaps are already NaN-masked upstream).

    Any noise-scale / calibration decision for the ensemble must be made on
    this training split, not load_test_set() - the Oct-Dec test set is held
    out for final evaluation only (see scripts/03_run_sr_model.py).

    Adds an 'is_ch_present' column (A_CH60_193_lag4 >= ch_threshold) so
    callers can separate CH-absent rows, where a/alpha perturbations are
    a no-op because A_CH*P_CH == 0.
    """
    return _load_split(1, 9, csv_path, ch_threshold)


def load_test_set(csv_path: str = _DEFAULT_DATA_PATH, ch_threshold: float = 0.01) -> pd.DataFrame:
    """
    Load the Oct-Dec held-out test set. Reserved for final evaluation only -
    use load_train_set() for sensitivity analysis / noise-scale calibration.
    """
    return _load_split(10, 12, csv_path, ch_threshold)


def load_full_series(csv_path: str = _DEFAULT_DATA_PATH, extra_cols: list = None) -> pd.DataFrame:
    """
    Load the full (all months, all years) hourly time series with 'speed' and
    the FEATURE_COLS columns, NaNs kept as-is (ICME periods leave the feature
    columns NaN but 'speed' intact, per the upstream ICME-masking convention).
    Intended for time-series / profile plots over an arbitrary date range,
    as opposed to load_test_set() which is restricted to the Oct-Dec test months.

    extra_cols lets callers pull in additional lag/latitude columns (e.g. for
    the structural latitude/delay ensembles in src/models/ensemble/structural.py)
    without a second read of the CSV.
    """
    needed = ["datetime", "speed"] + list(FEATURE_COLS.values()) + list(extra_cols or [])
    needed = list(dict.fromkeys(needed))  # de-dup, preserve order
    df = pd.read_csv(csv_path, usecols=needed)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)
