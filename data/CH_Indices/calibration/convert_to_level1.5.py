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

"Pointing correction"
def Pointing_correction(aia_map):
    pointing_tbl = get_pointing_table(
        "JSOC",
        time_range=(aia_map.date - 6*u.hour, aia_map.date + 6*u.hour)
    )
    aia_map_pt = aiapy.calibrate.update_pointing(aia_map, pointing_table=pointing_tbl)
    return aia_map_pt

"Registration"
def Registration(aia_map):
    aia_map_reg = aiapy.calibrate.register(
        aia_map,
        missing=np.nan,     # extrapolation: fill with NaN
        order=3,            # interpolation: bicubic        
        method='scipy'      # Rotation function to use: scipy
    )
    return aia_map_reg

"Degradation correction"
def Degradation_correction(aia_map):
    corr_tbl = get_correction_table("SSW")

    aia_map_cal = aiapy.calibrate.correct_degradation(
        aia_map,
        correction_table=corr_tbl
    )
    return aia_map_cal

"Exposure normalization"
def Exposure_normalization(aia_map):
    exp_time = aia_map.exposure_time
    aia_map_norm = aia_map / exp_time
    return aia_map_norm


def convert_to_level1_5(aia_map):
    """
    Convert SDO/AIA data from level 1 to level 1.5.
    """
    aia_map = Pointing_correction(aia_map)      # Step 1: Pointing correction
    aia_map = Registration(aia_map)             # Step 4: Registration
    aia_map = Degradation_correction(aia_map)   # Step 5: Degradation correction
    aia_map = Exposure_normalization(aia_map)   # Step 6: Exposure normalization

    return aia_map