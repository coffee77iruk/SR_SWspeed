"""
Convert SDO/AIA data from level 1 to level 1.5.

AIA data products provides by the JSOC are level 1 data products.  
This means that the images still include the roll angle of the satellite and each channel may have a slightly different pixel scale.  
Typically, before performing any sort of data analysis on AIA images, you will want to promote your AIA data from level 1 to level 1.5.

1. Pointing correction (aiapy.calibrate.update_pointing)  
2. Image respiking (aiapy.calibrate.respike)  
3. PSF deconvolution (aiapy.psf.deconvolve)  
4. Registration (aiapy.calibrate.register)  
5. Degradation correction (aiapy.calibrate.correct_degradation)  
6. Exposure normalization

In this code, we only use method 1, 4, 5, and 6.

Reference.
https://aiapy.readthedocs.io/en/stable/preparing_data.html

"""

import numpy as np
import astropy.units as u
import aiapy
from aiapy.calibrate.util import get_pointing_table
from aiapy.calibrate.util import get_correction_table

from sunpy.time import parse_time

# Remove the BLANK keyword
def clean_header(aia_map):
    if "BLANK" in aia_map.meta:
        del aia_map.meta["BLANK"]
    return aia_map

# Pointing correction
def Pointing_correction(aia_map, pointing_table=None):
    """
    We consider the satellite's attitude changes and movements to adjust the positioning of AIA images.

    pointing_table : pre-fetched table from get_pointing_table(), spanning a
    wide enough window to cover aia_map.date (update_pointing() searches the
    whole table for the matching row regardless of its size). Pass this in
    to skip a fresh JSOC network round-trip per call -- if omitted, falls
    back to fetching a narrow +-6h window for this map's own date, as before.
    """
    aia_map = clean_header(aia_map)  # Clean the header first
    if pointing_table is None:
        ref_date = parse_time(aia_map.date.isot)
        # select a lmsal or jsoc
        pointing_table = get_pointing_table(
            "jsoc",
            time_range=(ref_date - 6*u.hour, ref_date + 6*u.hour)
        )
    if aia_map.meta.get("SAT_ROT") is None:
        aia_map.meta["SAT_ROT"] = 0.0

    aia_map_pt = aiapy.calibrate.update_pointing(aia_map, pointing_table=pointing_table)
    return aia_map_pt

# Registration
def Registration(aia_map):
    """
    We rotate the AIA images to align the solar polar region to the top of the screen 
    and match the pixel size of each channel. 
    """
    aia_map = clean_header(aia_map)
    aia_map_reg = aiapy.calibrate.register(
        aia_map,
        missing=np.nan,     # fill the outer pixel spaces with NaN
        order=3,            # interpolation: bicubic        
        method='scipy'      # Rotation function to use: scipy
    )
    return aia_map_reg

# Degradation correction
def Degradation_correction(aia_map, correction_table=None):
    """
    We calibrate the degradation of AIA data to ensure
    that the physical brightness is consistent across different channels.

    correction_table : pre-fetched table from get_correction_table("SSW").
    This table isn't scoped to any particular date (get_correction_table
    takes no time_range), so it's identical on every call for a given SSW
    version -- pass one in to skip a redundant network fetch per call. If
    omitted, falls back to fetching it fresh, as before.
    """
    aia_map = clean_header(aia_map)
    if correction_table is None:
        correction_table = get_correction_table("SSW")
    aia_map_cal = aiapy.calibrate.correct_degradation(
        aia_map,
        correction_table=correction_table
    )
    return aia_map_cal

# Exposure normalization
def Exposure_normalization(aia_map):
    """
    We normalize the brightness of AIA images according to the exposure time.
    """
    aia_map = clean_header(aia_map)
    exp_time = aia_map.exposure_time
    if exp_time == 0 * u.s:
        return aia_map
    aia_map_norm = aia_map / exp_time
    #aia_map_norm.meta['BUNIT'] = 'DN / s'
    #aia_map_norm.meta['EXPCORR'] = True
    return aia_map_norm

# Main function to convert level 1 to level 1.5
def convert_to_level1_5(aia_map, pointing_table=None, correction_table=None):
    """
    Convert SDO/AIA data from level 1 to level 1.5.

    pointing_table/correction_table : optional pre-fetched tables (see
    Pointing_correction/Degradation_correction) to avoid a fresh network
    fetch on every call -- useful when converting many maps in a batch.
    """
    aia_map = Pointing_correction(aia_map, pointing_table=pointing_table)      # Step 1: Pointing correction
    aia_map = Registration(aia_map)                                           # Step 4: Registration
    aia_map = Degradation_correction(aia_map, correction_table=correction_table)  # Step 5: Degradation correction
    aia_map = Exposure_normalization(aia_map)                                  # Step 6: Exposure normalization

    return aia_map
