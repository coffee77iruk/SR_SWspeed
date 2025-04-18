"""
Run the conversion of SDO/AIA level 1 FITS files to level 1.5.

Convert SDO/AIA level 1 FITS files to level 1.5 and save them
with a progress bar that shows the number of files processed.

"""

import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import astropy.units as u
import sunpy
from sunpy.time import parse_time
from datetime import datetime
from aiapy.calibrate.util import get_correction_table
from aiapy.calibrate.util import get_pointing_table

from convert_to_level1_5 import strip_invalid_blank
from convert_to_level1_5 import convert_to_level1_5

def init_worker():
    import convert_to_level1_5
    convert_to_level1_5.set_correction_table(get_correction_table("JSOC"))
    
# Run the conversion in parallel using multiprocessing
def batch_worker(jobs_batch, pointing_cache):
    from convert_to_level1_5 import convert_to_level1_5, strip_invalid_blank
    import aiapy

    results = []
    for in_path, out_path in jobs_batch:
        try:
            aia_map = sunpy.map.Map(in_path)
            # calibrate by using the cached pointing table
            date_key = aia_map.date.isot[:10]
            pt_tbl   = pointing_cache[date_key]
            aia_map  = aiapy.calibrate.update_pointing(
                          aia_map, pointing_table=pt_tbl)

            # To skip the pointing correction step, set skip_pointing=True
            aia_map_new = convert_to_level1_5(aia_map, skip_pointing=True)
            strip_invalid_blank(aia_map_new)
            aia_map_new.save(out_path, overwrite=False)
            results.append((Path(in_path).name, True, ""))
        except Exception as e:
            results.append((Path(in_path).name, False, str(e)))
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Convert SDO/AIA data from level 1 to level 1.5."
    )
    parser.add_argument("--channel", type=str, required=True,
                        help="channel name (e.g, '193' or '193,211')")
    parser.add_argument("--start", type=str, required=True,
                        help="start date (e.g, 2012-01-01)")
    parser.add_argument("--end", type=str, required=True,
                        help="end date (e.g, 2024-12-31)")
    parser.add_argument("--file_directory", type=str, required=True,
                        help="directory containing level 1 FITS files (e.g, E:\Research\SR\input\CH_Indices\EUV_level1)")
    parser.add_argument("--save_directory", type=str, required=True,
                        help="directory to save a level 1.5 FITS files (e.g, E:\Research\SR\input\CH_Indices\EUV_level1.5)")
    parser.add_argument("--cores", type=int, default=4,
                        help="number of cores to use for processing")
    args = parser.parse_args()

    parent_dir = Path(args.file_directory)
    save_dir   = Path(args.save_directory)
    
    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()

    channels = [chan.strip() for chan in args.channel.split(',')]   # e.g., [193,211]
    batch_size = 20

    for chan in channels:
        all_jobs = []
        years = range(start_dt.year, end_dt.year + 1)
        for year in tqdm(years, desc=f"[{chan}] Collecting jobs by year"):
            source_dir  = parent_dir / chan / str(year)         # source directory
            destination_dir  = save_dir  / chan / str(year)     # destination directory
            destination_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist

            files = sorted(source_dir.glob("*.fits"))
            for file in tqdm(files, desc=f"[{chan} {year}] Checking files", leave=False):
                try:
                    file_date = datetime.strptime(file.stem.split(".")[2],      # Extract the date from the filename
                                               "%Y-%m-%dT%H%M%SZ")
                except Exception:
                    continue
                 
                if not (start_dt <= file_date <= end_dt):                       # Check if the file date is within the specified range
                    continue

                outfile = destination_dir / file.name.replace("lev1", "lev15")  # Check if the output file already exists
                if outfile.exists():
                    continue

                all_jobs.append((str(file), str(outfile)))                          # add a file and its destination path to the jobs list

        # Check if there are any jobs to process
        if not all_jobs:
            print(f"[{chan}] No files found.")
            continue

        unique_dates = {job[0].split(".")[2][:10] for job in all_jobs}
        pointing_cache = {}
        for d in tqdm(unique_dates, desc=f"[{chan}] Caching pointing tables"):
            tbl = get_pointing_table(
                "JSOC",
                time_range=(parse_time(d)-12*u.hour, 
                            parse_time(d)+12*u.hour)
            )
            pointing_cache[d] = tbl

        batches = [
            all_jobs[i:i+batch_size]
            for i in range(0, len(all_jobs), batch_size)
        ]

        ok = err = 0
        with ProcessPoolExecutor(max_workers=args.cores, initializer=init_worker) as executor:
            futures = [
                executor.submit(batch_worker, batch, pointing_cache)
                for batch in batches
            ]
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"EUV {chan}",
                               unit="batch"):
                for filename, success, msg in future.result():
                    if success: ok += 1
                    else:
                        err += 1
                        tqdm.write(f" ✖ {filename}: {msg}")

        print(f"[{chan}] done ➜ OK:{ok}  ERR:{err}")


if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices\calibration
# python run_convert_to_level1_5.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --file_directory "E:\Research\SR\input\CH_Indices\EUV_level1" --save_directory "E:\Research\SR\input\CH_Indices\EUV_level1.5" --cores 4