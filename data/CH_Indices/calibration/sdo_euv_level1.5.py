
import os
import argparse
from pathlib import Path
from datetime import datetime
from sunpy.time import parse_time
from tqdm import tqdm

import sunpy
from convert_to_level1_5 import convert_to_level1_5

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
    args = parser.parse_args()

    Parent_dir = args.file_directory
    save_dir = args.save_directory
    
    start_dt = parse_time(args.start).to_datetime()
    end_dt = parse_time(args.end).to_datetime()

    channels = [chan.strip() for chan in args.channel.split(',')]   # e.g., [193,211]
    for chan in tqdm(channels):
        for year in tqdm(range(start_dt.year, end_dt.year + 1)):
            # make lists of level 1 FITS files
            dir_level1 = os.path.join(Parent_dir, chan, str(year))
            list_level1 = os.listdir(dir_level1)
            # make directory to save level 1.5 FITS files
            dir_level1_5 = os.path.join(save_dir, chan, str(year))
            os.makedirs(dir_level1_5, exist_ok=True)

            for file in tqdm(list_level1):
                file_date = file.split('.')[2]
                file_date = datetime.strptime(file_date, "%Y-%m-%dT%H%M%SZ")
                if file_date > start_dt and file_date < end_dt:
                    # convert level 1 to level 1.5
                    aia_map = sunpy.map.Map(os.path.join(dir_level1, file))
                    aia_map_new = convert_to_level1_5(aia_map)
                    # save level 1.5 FITS files
                    os.path.join(dir_level1_5)
                    aia_map_new.save(os.path.join(dir_level1_5, file), overwrite=True)
                else:
                    continue
            print(f"Complete channel {chan} for year {year}. The results are saved at {dir_level1_5}.")

if __name__ == '__main__':
    main()


# To run this script, you can use the command line as follows:
# activate venv
# cd Research\SR_SWspeed\data\CH_Indices\calibration
# python sdo_euv_level1.5.py --channel "193,211" --start "2012-01-01" --end "2024-12-31" --file_directory "E:\Research\SR\input\CH_Indices\EUV_level1" --save_directory "E:\Research\SR\input\CH_Indices\EUV_level1.5"