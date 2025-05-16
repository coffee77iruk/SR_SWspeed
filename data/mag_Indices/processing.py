"""
To get a mag_Indices such as expansion factor, angular distance for CH area, squashing factor.

These are our function to calculate solar parameters:
1. compute expansion_factor
2. compute coronal_hole_dist
3. compute squashing_factor

"""
import numpy as np
import pandas as pd
from astropy.time import Time

def jd2dt(jd_series, bias=2440000.00):
    """
    Convert Julian date to datetime.
    """
    return Time(jd_series + bias, format="jd").to_datetime()
    
def get_mag_indices(dat_file) -> pd.DataFrame:
    """
    Get a WSA table from a file.
    """
    jul_bias = 2440000.00

    # Read the WSA data
    wsa_df = pd.read_csv(dat_file, comment='#', sep=r'\s+', engine='python')

    wsa_df["departure_time"] = jd2dt(wsa_df["parcel_depart_time"], jul_bias)
    wsa_df["arrival_time"]   = jd2dt(wsa_df["juldate"], jul_bias)

    # set the time range to 12 hours 
    hr = wsa_df["arrival_time"].dt.hour
    mask_noon = hr.between(11, 12, inclusive='both')        # 11‒13 h
    mask_midnight = (hr >= 23) | (hr <= 1)                  # 23‒01 h

    midnight = wsa_df['arrival_time'].dt.floor('D')
    masked_arrival_dt = np.select(condlist=[mask_noon, mask_midnight],
                                  choicelist=[midnight + pd.Timedelta(hours=12), midnight],
                                  default=pd.NaT)
    wsa_df['masked_arrival_time'] = pd.to_datetime(masked_arrival_dt)\
                                .strftime('%Y-%m-%dT%H:%M:%S')

    # Create a timeline DataFrame with 12-hour intervals
    # 2012-01-01 00:00:00 to 2024-12-31 12:00:00
    timeline = pd.DataFrame(
        {"datetime": pd.date_range("2012-01-01", "2024-12-31 12:00", freq="12h")
                 .strftime('%Y-%m-%dT%H:%M:%S')}
    )
    # Merge the WSA data with the timeline
    merged = timeline.merge(wsa_df, left_on="datetime", right_on="masked_arrival_time", how="left")

    avg_cols   = ["expansion_factor", "coronal_hole_dist", "squashing_factor"]
    agg = {**{c: "mean" for c in avg_cols},
           **{c: "first" for c in merged.columns.difference(avg_cols + ["datetime"])}}

    mag_df = (
        merged.groupby("datetime", as_index=False).agg(agg)
              [["datetime", "departure_time", "arrival_time", *avg_cols]]
    )

    mag_df[["departure_time", "arrival_time"]] = mag_df[
        ["departure_time", "arrival_time"]
    ].apply(pd.to_datetime, errors="coerce")

    for col in ["departure_time", "arrival_time"]:
        mag_df[col] = (
            mag_df[col]
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .fillna(pd.NaT)
        )

    return mag_df
