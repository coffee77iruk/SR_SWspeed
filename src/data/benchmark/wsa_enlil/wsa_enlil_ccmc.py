"""
Preprocessing of the WSA-ENLIL data for comparison with the SWspeed data.

"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from .cr_data import fetch_cr_table, shift_cr_dates, filter_cr_by_months

# file path
URL = "https://space.umd.edu/pm/crn/"
# Resolved relative to this file (not the caller's cwd), since this data lives
# under the top-level data/ directory rather than alongside src/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
enlil_file_path = str(_REPO_ROOT / "data" / "external" / "wsa_enlil")

def load_enlil_df(filepath: str) -> pd.DataFrame:
    start_dt = None
    colnames = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            if "Start Date, time:" in line:
                m = re.search(
                    r"(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})",
                    line
                )
                if not m:
                    raise ValueError(f"Cannot parse Start Date line: {line}")
                start_dt = pd.to_datetime(
                    f"{m.group(1)} {m.group(2)}",
                    format="%Y/%m/%d %H:%M:%S"
                )

            if line.startswith("#"):
                tokens = line.lstrip("#").strip().split()
                if tokens and tokens[0] == "Time":
                    colnames = tokens
                continue

            if line:
                break

    if start_dt is None:
        raise ValueError("Start Date, time not found in header.")
    if colnames is None:
        raise ValueError("Column name line '# Time ...' not found in header.")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=None,
        names=colnames
    )

    df["datetime_enlil"] = start_dt + pd.to_timedelta(df["Time"], unit="D")

    return df

def hourly_enlil_df(_df):
    df = _df.sort_values("datetime_enlil").reset_index(drop=True)

    start_hr = df["datetime_enlil"].min().ceil("h")
    end_hr   = df["datetime_enlil"].max().floor("h")

    time_df = pd.DataFrame({
        "datetime": pd.date_range(start=start_hr, end=end_hr, freq="h")
    })

    hourly_df = pd.merge_asof(
        time_df,
        df[["datetime_enlil", "V"]].rename(columns={"V": "speed"}),
        left_on="datetime",
        right_on="datetime_enlil",
        direction="nearest",
        tolerance=pd.Timedelta("5min")
    )

    return hourly_df


def WSA_ENLIL(shift_days: int = 4):
    CR_df = fetch_cr_table(URL)
    CR_df = shift_cr_dates(CR_df, shift_days)
    CR_df = filter_cr_by_months(CR_df)

    CR_df['Year'] = CR_df['Start Date'].dt.year
    idx = CR_df.groupby('Year')['Start Date'].idxmin()
    first_cr_list = CR_df.loc[idx, 'Carrington Rotation Number'].tolist()

    CR_list_sorted = sorted(os.listdir(enlil_file_path))

    enlil_df = None

    for fname in CR_list_sorted:
        if not fname.endswith(".txt"):
            continue

        m = re.search(r"(\d{4})", fname)
        if not m:
            continue

        cr_num = int(m.group(1))
        filepath = os.path.join(enlil_file_path, fname)

        df = load_enlil_df(filepath)
        df = hourly_enlil_df(df)
        year = df["datetime"].dt.year.min()

        if cr_num in first_cr_list:    
            piece = df[df["datetime"] >= pd.Timestamp(year, 10, 1, 0, 0, 0)].copy()
            #print(f"{filepath}  --> first CR of year")
        else:
            boundary = df['datetime'].iloc[0] + timedelta(days=shift_days)
            piece = df[(df["datetime"] >= boundary) & (df["datetime"] < pd.Timestamp(year+1, 1, 1, 0, 0, 0))].copy()
            #print(f"{filepath}  --> normal CR (override from {boundary})")

        piece = piece.set_index("datetime").sort_index()

        if enlil_df is None:
            enlil_df = piece
        else:
            enlil_df = enlil_df.combine_first(piece)
            enlil_df.update(piece)

    enlil_df = enlil_df.reset_index()

    return enlil_df
