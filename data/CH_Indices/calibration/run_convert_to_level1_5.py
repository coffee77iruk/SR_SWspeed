"""
Convert SDO/AIA level 1 FITS files to level 1.5 and save them
with a progress bar that 표시 currently processed file.

"""

import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import sunpy
from sunpy.time import parse_time
from datetime import datetime

from convert_to_level1_5 import convert_to_level1_5

def strip_invalid_blank(aia_map):
    """
    Remove the BLANK keyword when BITPIX < 0 (float data),
    to avoid astropy VerifyWarning.
    """
    if aia_map.meta.get("BITPIX", 0) < 0 and "BLANK" in aia_map.meta:
        aia_map.meta.pop("BLANK") 

# Run the conversion in parallel using multiprocessing
def worker(in_path: str, out_path: str) -> tuple[str, bool, str]:
    """
    Parameters
    ----------
    in_path  : original level 1 FITS
    out_path : destination file for level 1.5 FITS
 
    Returns (filename, ok_flag, message)
    """
    try:
        aia_map = sunpy.map.Map(in_path)
        aia_map_new = convert_to_level1_5(aia_map)
        strip_invalid_blank(aia_map_new)
        aia_map_new.save(out_path, overwrite=False)
        return (Path(in_path).name, True, "")
    except Exception as exc:
        return (Path(in_path).name, False, str(exc))

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
                        help="directory containing level 1 FITS files")
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

    for chan in channels:
        for year in range(start_dt.year, end_dt.year + 1):
            src_dir  = parent_dir / chan / str(year)    # source directory
            dst_dir  = save_dir  / chan / str(year)     # destination directory
            dst_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(src_dir.glob("*.fits"))
            jobs  = []

            for file in files:
                 # Extract the date from the filename
                 try:
                     file_date = datetime.strptime(file.stem.split(".")[2],
                                               "%Y-%m-%dT%H%M%SZ")
                 except Exception:
                     continue
                 # Check if the file date is within the specified range
                 if not (start_dt <= file_date <= end_dt):
                     continue
                 # Check if the output file already exists
                 outfile = dst_dir / file.name.replace("lev1", "lev15")
                 if outfile.exists():
                     continue
                 jobs.append((str(file), str(outfile)))

            # jobs: list of tuples (input_path, output_path)
            if not jobs:
                print(f"[{chan}] {year}  No files found.")
                continue

            # Process the jobs in parallel
            ok = err = 0
            with ProcessPoolExecutor(max_workers=args.cores) as executor:
                futures = [executor.submit(worker, job[0], job[1]) for job in jobs]
                # Use tqdm to show progress bar
                for future in tqdm(as_completed(futures), 
                                total=len(futures),
                                desc=f"EUV {chan} | year={year}", 
                                unit="file"):
                    filename, success, message = future.result()
                    if success:
                        ok += 1
                    else:
                        err += 1
                        tqdm.write(f"    ✖ {filename}: {message}")
 
            print(f"[{chan}] {year} done ➜ OK:{ok}  ERR:{err}")
 
    print("All conversions finished.")

if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices\calibration
# python run_convert_to_level1_5.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --file_directory "E:\Research\SR\input\CH_Indices\EUV_level1" --save_directory "E:\Research\SR\input\CH_Indices\EUV_level1.5"