"""
Run the conversion of SDO/AIA level 1 FITS files to level 1.5.

Convert SDO/AIA level 1 FITS files to level 1.5 and save them
with a progress bar that shows the number of files processed.

"""

import argparse
from pathlib import Path
from tqdm import tqdm

import sunpy
from sunpy.time import parse_time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from convert_to_level1_5 import convert_to_level1_5
import warnings

def process_and_save(infile: str, outfile: str):
    """
    Save a FITS file after converting to level 1.5
    """
    aia_map = sunpy.map.Map(infile)
    aia_map_new = convert_to_level1_5(aia_map)
    
    aia_map_new.save(outfile)

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
                        help="directory to save a level 1.5 FITS files (e.g, D:\Research_data\EUV)")
    parser.add_argument("--cores", type=int, default=4,
                        help="number of cores to use for processing")
    args = parser.parse_args()

    parent_dir = Path(args.file_directory)
    save_dir   = Path(args.save_directory)
    
    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()

    channels = [chan.strip() for chan in args.channel.split(',')]   # e.g., [193,211]
    years = range(start_dt.year, end_dt.year + 1)

    for chan in channels:
        for year in years:
            source_dir = parent_dir / str(chan) / str(year)
            destination_dir = save_dir / str(chan)/ str(year)
            destination_dir.mkdir(parents=True, exist_ok=True)

            destination_files = []
            for file in sorted(source_dir.glob("*.fits")):
                try:
                    file_dt = datetime.strptime(file.stem.split(".")[2],
                                                "%Y-%m-%dT%H%M%SZ")
                except Exception:
                    continue
                if not (start_dt <= file_dt <= end_dt):
                    continue

                outpath = destination_dir / file.name.replace("lev1", "lev1_5")
                if outpath.exists():
                    continue

                destination_files.append((str(file), str(outpath)))

            if not destination_files:
                continue

            with ProcessPoolExecutor(max_workers=args.cores) as executor:
                futures = {
                    executor.submit(process_and_save, inp, outp): (inp, outp)
                    for inp, outp in destination_files
                }

                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"EUV {chan} | year={year}",
                                   unit="file"):
                    inp, outp = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        tqdm.write(f"[ERROR] {Path(inp).name} -> {e}")

    print("All conversions finished.")

if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:

# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices\calibration
# python run_convert_to_level1_5.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --file_directory "E:\Research\SR\input\CH_Indices\EUV_level1" --save_directory "D:\Research_data\EUV"