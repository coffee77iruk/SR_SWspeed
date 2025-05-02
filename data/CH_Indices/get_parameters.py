"""
Run a extracting of CH_Indics values.

...

"""

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

import warnings

def get_parameter(file):
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

# processing function
def process_dt(dt, chan, source_dir):
    prefix = dt.strftime('%Y-%m-%dT%H')
    pattern = f"aia.lev1_5_euv_12s.{prefix}*Z.{chan}.image_lev1_5.fits"
    matches = list(source_dir.glob(pattern))
    if matches:
        fpath = matches[0]
    else:
        # dummy path to trigger nan in get_parameter
        fpath = source_dir / 'file_not_found.fits'
    return dt, *get_parameter(fpath)

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

    for chan in channels:
        for year in years:
            source_dir = base_dir / str(chan) / str(year)
            destination_dir = save_dir / str(chan)
            destination_dir.mkdir(parents=True, exist_ok=True)
            save_file = destination_dir / f"CH_Indics_{chan}.csv"   # save_file is a csv file.

            # --- resume logic ---
            if save_file.exists():
                df = pd.read_csv(save_file)
                df.columns = df.columns.str.strip()
                if 'datetime' not in df.columns:
                    alt = [c for c in df.columns if c.strip().lower() == 'datetime']
                    if alt:
                        df.rename(columns={alt[0]: 'datetime'}, inplace=True)
                    else:
                        raise KeyError(f"No 'datetime' column; existing columns: {df.columns.tolist()}")
                df['datetime'] = pd.to_datetime(
                    df['datetime'],
                    format="%Y-%m-%d_%H",
                    errors='coerce'
                )
                processed = set(df['datetime'].dt.to_pydatetime())
            else:
                save_file.write_text("datetime,A_CH,P_CH30,P_CH90\n")
                processed = set()

            # build expected datetime list for this year
            year_start = max(start_dt, datetime(year, 1, 1, 0, 0))
            year_end   = min(end_dt,   datetime(year, 12, 31, 23, 59, 59))
            current = year_start
            dt_list = []
            while current <= year_end:
                dt_list.append(current)
                current += timedelta(hours=args.cadence)

            # filter out already processed
            to_do = [dt for dt in dt_list if dt not in processed]

            def write_line(dt, a_ch, p_ch30, p_ch90):
                # datetime from map Time or fallback dt
                if isinstance(a_ch, tuple) and a_ch[0] is not None:
                    time_str = a_ch[0].isot.split('.')[0]
                else:
                    time_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
                # values only
                a_val = a_ch[1] if isinstance(a_ch, tuple) else a_ch
                p30_val = p_ch30[1] if isinstance(p_ch30, tuple) else p_ch30
                p90_val = p_ch90[1] if isinstance(p_ch90, tuple) else p_ch90
                line = f"{time_str},{a_val},{p30_val},{p90_val}\n"
                save_file.open('a').write(line)

            # run
            if args.cores > 1 and to_do:
                worker = partial(process_dt, chan=chan, source_dir=source_dir)
                with Pool(args.cores) as pool:
                    for dt, a_ch, p_ch30, p_ch90 in tqdm(
                        pool.imap(worker, to_do),
                        total=len(to_do),
                        desc=f"Channel {chan} Year {year}", unit="time"
                    ):
                        write_line(dt, a_ch, p_ch30, p_ch90)
            else:
                for dt, a_ch, p_ch30, p_ch90 in tqdm(
                    (process_dt(dt, chan, source_dir) for dt in to_do),
                    total=len(to_do),
                    desc=f"Channel {chan} Year {year}", unit="time"
                ):
                    write_line(dt, a_ch, p_ch30, p_ch90)

    print("All running is finished.")

if __name__ == "__main__":
    mp.freeze_support()
    main()

# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices
# python get_parameters.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --cadence 12 --base_dir "D:\Data\EUV" --save_dir "D:\Data\EUV" --cores 4