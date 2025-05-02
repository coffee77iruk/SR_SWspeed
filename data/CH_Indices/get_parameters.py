"""
Extract CH_Indics values from SDO/AIA level 1.5 FITS files and save to CSV.

Usage:
  python get_parameters.py \
    --channel "193,211" \
    --start "2012-01-01" \
    --end "2024-12-31" \
    --cadence 12 \
    --base_dir "D:/Data/EUV" \
    --save_dir "D:/Data/EUV" \
    --cores 4

"""

import os
import numpy as np
import pandas as pd

import argparse
from pathlib import Path
from tqdm import tqdm

from sunpy.time import parse_time
from datetime import datetime, timedelta

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from processing import get_A_CH, get_P_CH


def get_last_processed(save_file: Path, fmt: str = '%Y-%m-%dT%H:%M:%S'):
    """
    Read the last line in `save_file` and parse its datetime.

    """
    if not save_file.exists() or save_file.stat().st_size == 0:
        return None
    with open(save_file, 'rb') as f:
        try:
            # Seek from end to find start of last line
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            # File too small, rewind to start
            f.seek(0)
        # Read and decode the last line
        last_line = f.readline().decode().strip()

    if not last_line:
        return None     # No valid line found
    
    # The timestamp is the first CSV field
    last_dt_str = last_line.split(',')[0]
    try:
        return datetime.strptime(last_dt_str, fmt)
    except ValueError:
        # Format mismatch or parse error
        return None
    

def process_dt(dt: datetime, chan: str, source_dir: Path):
    """
    For a given datetime `dt` and channel name, find the corresponding FITS file
    and extract CH indices.

    """
    prefix = dt.strftime('%Y-%m-%dT%H')
    pattern = f"aia.lev1_5_euv_12s.{prefix}*Z.{chan}.image_lev1_5.fits"
    matches = list(source_dir.glob(pattern))
    fpath = matches[0] if matches else source_dir / 'file_not_found.fits'
    return dt, fpath, *get_parameter(fpath)


def get_parameter(file: Path):
    """
    Call processing functions to compute CH indices for the given FITS file.

    """
    if file.exists():
        try:
            t_a, a_ch = get_A_CH(file)
            t_p30, p_ch30 = get_P_CH(file, lon=10, lat=30)
            t_p90, p_ch90 = get_P_CH(file, lon=10, lat=90)
            return (t_a, a_ch), (t_p30, p_ch30), (t_p90, p_ch90)
        except Exception:
            return np.nan, np.nan, np.nan
    else:
        return np.nan, np.nan, np.nan


def write_line(save_file: Path, dt: datetime, a_ch, p_ch30, p_ch90):
    """
    Append a CSV line for the given datetime and CH indices values.
    Always uses `dt` as the timestamp for consistency.

    """
    time_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
    a_val = a_ch[1] if isinstance(a_ch, tuple) else a_ch
    p30_val = p_ch30[1] if isinstance(p_ch30, tuple) else p_ch30
    p90_val = p_ch90[1] if isinstance(p_ch90, tuple) else p_ch90
    line = f"{time_str},{a_val},{p30_val},{p90_val}\n"
    with open(save_file, 'a') as f:
        f.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the CH_Indics values from FITS files."
    )
    parser.add_argument("--channel", type=str, required=True,
                        help="channel names (e.g., '193' or '193,211')")
    parser.add_argument("--start", type=str, required=True,
                        help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="end date (YYYY-MM-DD)")
    parser.add_argument("--cadence", type=int, default=12,
                        help="cadence in hours (e.g., 12 or 24)")
    parser.add_argument("--base_dir", type=str, default=r"D:\\Data\\EUV",
                        help="parent folder with FITS files")
    parser.add_argument("--save_dir", type=str, default=r"D:\\Data\\EUV",
                        help="folder to save results CSV")
    parser.add_argument("--cores", type=int, default=1,
                        help="number of cores to use for processing")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    
    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()

    channels = [chan.strip() for chan in args.channel.split(',')]   # e.g., [193,211]
    years = range(start_dt.year, end_dt.year + 1)
    fmt = '%Y-%m-%dT%H:%M:%S'

    for chan in channels:
        save_file = save_dir / str(chan) / f"CH_Indics_{chan}.csv"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        # Initialize file with header if empty or new
        if not save_file.exists() or save_file.stat().st_size == 0:
            save_file.write_text("datetime,A_CH,P_CH30,P_CH90\n")

        for year in years:
            source_dir = base_dir / str(chan) / str(year)

            # Define the processing window for this year
            year_start = max(start_dt, datetime(year, 1, 1, 0, 0, 0))
            year_end = min(end_dt, datetime(year, 12, 31, 23, 59, 59))
            last_dt = get_last_processed(save_file, fmt)

            # If we've already processed beyond the year start, pick up from there
            if last_dt and last_dt + timedelta(hours=args.cadence) > year_start:
                current = last_dt + timedelta(hours=args.cadence)
            else:
                current = year_start

            # Build list of datetimes at the specified cadence
            dt_list = []
            while current <= year_end:
                dt_list.append(current)
                current += timedelta(hours=args.cadence)

            if not dt_list:
                continue

            desc = f"Wavelength {chan} Year {year}"
            # Partially apply fixed arguments for worker function
            worker = partial(process_dt, chan=chan, source_dir=source_dir)

            # Parallel or serial processing based on core count
            if args.cores > 1:
                with Pool(args.cores) as pool:
                    pbar = tqdm(pool.imap(worker, dt_list), total=len(dt_list), unit="step")
                    for dt, fpath, a_ch, p_ch30, p_ch90 in pbar:
                        pbar.set_description(f"{desc} | {fpath.name.split('.')[2]}")
                        write_line(save_file, dt, a_ch, p_ch30, p_ch90)
            else:
                pbar = tqdm((process_dt(dt, chan, source_dir) for dt in dt_list),
                            total=len(dt_list), unit="step")
                for dt, fpath, a_ch, p_ch30, p_ch90 in pbar:
                    pbar.set_description(f"{desc} | {fpath.name.split('.')[2]}")
                    write_line(save_file, dt, a_ch, p_ch30, p_ch90)

        print(f"Channel {chan} processing complete.")

    print("All running finished.")


if __name__ == "__main__":
    mp.freeze_support()
    main()


# To run this script, you can use the command line as follows:

# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices
# python get_parameters.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --cadence 12 --base_dir "D:\Data\EUV" --save_dir "D:\Data\EUV" --cores 4