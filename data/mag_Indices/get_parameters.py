"""
Extract mag_Indics values and save to CSV.

"""

import os

from processing import get_mag_indices

def main():
    base_path = "E:\Research\SR\input\mag_Indices"
    R5_0_file_path  = "\R5_0\PREDSOLARWIND\GONGZfield_line1R000_R5.0.dat"
    R21_5_file_path = "\R21_5\PREDSOLARWIND\GONGZfield_line1R000.dat"

    R5_0_df  = get_mag_indices(base_path + R5_0_file_path)
    R21_5_df = get_mag_indices(base_path + R21_5_file_path)

    # Save the DataFrame to a CSV file
    os.makedirs(base_path, exist_ok=True) 

    R5_0_df.to_csv(os.path.join(base_path, "mag_indices_R5_0.csv"),  index=False)
    R21_5_df.to_csv(os.path.join(base_path, "mag_indices_R21_5.csv"), index=False)

if __name__ == "__main__":
    main()


# To run this script, you can use the command line as follows:

# conda activate venv
# cd Research\SR_SWspeed\data\mag_Indices
# python get_parameters.py