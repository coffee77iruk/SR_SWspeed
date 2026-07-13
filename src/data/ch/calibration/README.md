We only use convert_to_level1_5.py to extract coronal hole parameters directly from AIA data.
The scripts run_convert_to_level1_5.py and run_convert_to_level1_5_image.py are not required for this workflow, as they only generate Level 1.5 FITS files and images separately.
Since our pipeline uses Level 1.5 data only for parameter extraction, there is no need to run these additional scripts.

Images were corrected for spacecraft pointing using the JSOC MPT,
while degradation correction was performed using the SSW AIA response table.