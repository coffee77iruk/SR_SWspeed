"""
Convert SDO/AIA level 1 FITS files to level 1.5 and save them
with a progress bar that 표시 currently processed file.

"""

import argparse
from pathlib import Path
from datetime import datetime
from sunpy.time import parse_time
from tqdm import tqdm

import sunpy
from convert_to_level1_5 import convert_to_level1_5

def strip_invalid_blank(aia_map):
    """
    Remove the BLANK keyword when BITPIX < 0 (float data),
    to avoid astropy VerifyWarning.
    """
    if aia_map.meta.get("BITPIX", 0) < 0 and "BLANK" in aia_map.meta:
        aia_map.meta.pop("BLANK") 

def main() -> None:
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

            files = [f for f in src_dir.glob("*.fits")]
            if not files:
                print(f"[{chan}] {year}: no FITS files in {src_dir}")
                continue

            success = fail = skipped = 0

            with tqdm(files, desc=f"EUV {chan}, year={year}", unit="file") as pbar:
                for file in pbar:
                    pbar.set_postfix(file=file.name[:60])
                    try:
                        file_date = datetime.strptime(file.stem.split(".")[2],
                                                      "%Y-%m-%dT%H%M%SZ")
                    except (IndexError, ValueError):
                        skipped += 1
                        continue

                    if not (start_dt <= file_date <= end_dt):
                        skipped += 1
                        continue

                    outfile = dst_dir / file.name.replace("lev1", "lev15")
                    if outfile.exists():
                        skipped += 1
                        continue

                    try:
                        aia_map = sunpy.map.Map(file)
                        aia_map_new = convert_to_level1_5(aia_map)
                        strip_invalid_blank(aia_map_new)
                        aia_map_new.save(outfile, overwrite=False)
                        success += 1
                    except Exception as e:
                        pbar.write(f"[{chan}] {file.name} failed: {e}")
                        fail += 1
                pbar.write(
                    f"[{chan}] {year}  ✔:{success}  ✖:{fail}  ➜skipped:{skipped}"
                )
    print("All conversions finished.")

if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# conda activate venv
# cd Research\SR_SWspeed\data\CH_Indices\calibration
# python sdo_euv_level1.5.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --file_directory "E:\Research\SR\input\CH_Indices\EUV_level1" --save_directory "E:\Research\SR\input\CH_Indices\EUV_level1.5"