import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# SunPy fallback version
try: 
    from sunpy.coordinates.sun import carrington_rotation_time
    HAS_SUNPY = True
except ImportError:
    HAS_SUNPY = False

URL = "https://space.umd.edu/pm/crn/"

def fetch_cr_table_from_sunpy(cr_start: int = 2096, cr_end: int = 2295) -> pd.DataFrame:
    """
    Return the start/end time of the Carrington rotation using SunPy.

    Parameters
    ----------
    cr_start : int
        Start Carrington rotation number (default: 2096, ~2010 May)
    cr_end : int
        End Carrington rotation number (default: 2295, ~2024 Dec)
    """
    
    cr_numbers = np.arange(cr_start, cr_end + 2)  
    start_times = [carrington_rotation_time(cr).to_datetime() for cr in cr_numbers]

    data = []
    for i, cr in enumerate(cr_numbers[:-1]):  
        data.append({
            "Carrington Rotation Number": cr,
            "Start Date": pd.Timestamp(start_times[i]),
            "End Date":   pd.Timestamp(start_times[i + 1]),
        })

    df = pd.DataFrame(data)
    df = df.sort_values("Carrington Rotation Number", ascending=True).reset_index(drop=True)
    return df


def fetch_cr_table(url: str = URL, cr_start: int = 2096, cr_end: int = 2295) -> pd.DataFrame:
    """
    Crawl the UMD CRN webpage and return a DataFrame containing 
    the start and end times for each Carrington rotation.

    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        start_idx = next(i for i, line in enumerate(lines) if line == "Car Rot")
        
        data = []
        i = start_idx + 4  # Car Rot, Start Time, Stop Time, Links

        while i + 6 < len(lines):
            line = lines[i]

            if line.startswith("** No data"):
                break

            car_rot = int(line)
            start_date, start_time = lines[i + 1], lines[i + 2]
            end_date,   end_time   = lines[i + 3], lines[i + 4]

            start_str = f"{start_date} {start_time}"  # e.g.: "2025 Nov 30 0117"
            end_str   = f"{end_date} {end_time}"      # e.g.: "2025 Dec 27 0903"

            data.append((car_rot, start_str, end_str))

            # car_rot, start_date, start_time, end_date, end_time, Plot, List
            i += 7


        df = pd.DataFrame(data, columns=["Carrington Rotation Number", "Start Date", "End Date"])
        df["Start Date"] = pd.to_datetime(df["Start Date"], format="%Y %b %d %H%M")
        df["End Date"]   = pd.to_datetime(df["End Date"],   format="%Y %b %d %H%M")
        
        df = df.sort_values("Carrington Rotation Number", ascending=True).reset_index(drop=True)
        return df
    
    except Exception as e:
        print(f"[WARNING] Fail to connect UMD site")
        return fetch_cr_table_from_sunpy(cr_start=cr_start, cr_end=cr_end)


def shift_cr_dates(cr_df, shift_days: int):
    """Shift Start Date and End Date by a given number of days."""
    
    delta = pd.to_timedelta(shift_days, unit="D")
    
    df = cr_df.copy()
    df["Start Date"] = df["Start Date"] + delta
    df["End Date"]   = df["End Date"] + delta
    
    return df


def filter_cr_by_months(cr_df, start_year=2010, end_year=2024, target_months=[10, 11, 12]):
    """
    Filter Carrington rotation rows where either Start Date or End Date falls
    within the given year range AND within the specified target months.

    """

    start_mask = (
        cr_df["Start Date"].dt.year.between(start_year, end_year)
        & cr_df["Start Date"].dt.month.isin(target_months)
    )

    end_mask = (
        cr_df["End Date"].dt.year.between(start_year, end_year)
        & cr_df["End Date"].dt.month.isin(target_months)
    )

    mask = start_mask | end_mask

    return cr_df[mask].reset_index(drop=True)
