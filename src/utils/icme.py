"""
ICME (Interplanetary Coronal Mass Ejection) event utilities.

Functions
---------
fetch_icme_events  : Download and parse the Caltech ACE ICME catalog.
mask_icme_events   : Flag ICME periods (and their lagged persistence windows) in a DataFrame.
make_icme_mask     : Build a boolean numpy array marking ICME time indices.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

ICME_URL = "https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"


@lru_cache(maxsize=None)
def fetch_icme_events(url: str = ICME_URL,
                      year_start: int = 2010,
                      year_end: int = 2024) -> list:
    """
    Fetch ICME event list from the Caltech ACE ICME catalog.

    Returns a list of (start, end) tuples where start and end are
    pandas Timestamp objects.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"ICME catalog request failed: {response.status_code}")

    soup  = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    icme_entire_table = pd.read_html(StringIO(str(table)))[0]

    icme_sub = icme_entire_table.iloc[:, [1, 2, 11, 12]].copy()
    icme_sub.columns = ['ICME_start', 'ICME_end', 'ICME_mean', 'ICME_max']
    icme_sub = icme_sub.dropna(subset=['ICME_start', 'ICME_end'])

    icme_sub['ICME_start'] = pd.to_datetime(
        icme_sub['ICME_start'], errors='coerce', format='%Y/%m/%d %H%M'
    )
    icme_sub['ICME_end'] = pd.to_datetime(
        icme_sub['ICME_end'], errors='coerce', format='%Y/%m/%d %H%M'
    )

    mask = (
        (icme_sub['ICME_start'].dt.year >= year_start) &
        (icme_sub['ICME_start'].dt.year <= year_end) &
        (icme_sub['ICME_end'].dt.year >= year_start) &
        (icme_sub['ICME_end'].dt.year <= year_end)
    )
    icme_df = icme_sub[mask].reset_index(drop=True)

    return list(zip(icme_df['ICME_start'], icme_df['ICME_end']))


def mask_icme_events(df: pd.DataFrame,
                     icme_intervals: list,
                     persistence_shifts: dict) -> pd.DataFrame:
    """
    Flag rows in df that fall within an ICME interval or within a
    persistence-shifted window of an ICME interval.

    Adds an 'is_ICME' boolean column to the returned DataFrame.
    """
    df = df.copy()
    df['is_ICME'] = False

    for start, end in icme_intervals:
        mask_main = df['datetime'].between(start, end)
        df.loc[mask_main, 'is_ICME'] = True

        for shift in persistence_shifts.values():
            start_s = start + pd.Timedelta(hours=shift)
            end_s   = end   + pd.Timedelta(hours=shift)
            df.loc[df['datetime'].between(start_s, end_s), 'is_ICME'] = True

    return df.reset_index(drop=True)


def make_icme_mask(time_all: pd.Series, icme_intervals: list) -> np.ndarray:
    """
    Build a boolean array the same length as time_all where True indicates
    the time point falls within an ICME interval.
    """
    icme_mask = np.zeros(len(time_all), dtype=bool)
    t = pd.to_datetime(time_all).to_numpy()

    for start, end in icme_intervals:
        icme_mask |= (t >= np.datetime64(start)) & (t <= np.datetime64(end))

    return icme_mask
