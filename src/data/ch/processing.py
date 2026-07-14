"""
To get a CH_Indices such as A_CH, P_CH, theta(latitude).

These are our function to calculate solar parameters:
1. compute_A_CH
2. compute_P_CH
3. compute_theta

"""

import numpy as np
from matplotlib.path import Path

import astropy.units as u                   # process unit
from astropy.coordinates import SkyCoord    # celestial coordinate system
from astropy.time import TimeDelta          # time processing

from sunpy.map import Map
from sunpy.coordinates import frames        # celestial coordinate system
from sunpy.net import attrs as a            # search condition definition
from sunpy.net import hek                   # solar event search

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

# A_CH parameter
def get_A_CH(aia_map, lon_limits=[7.5], lat_limits=[30, 60, 90]):
    """
    Read the given sunpy.map.Map variable, search for CH events using HEK,
    and calculate A_CH using the number of internal pixels within each
    central meridional region (±lon, ±lat).

    Return: (aia_map.date, A_CH) where A_CH is a flat list ordered lon-major,
    lat-minor: [lon_limits[0]xlat_limits[0], lon_limits[0]xlat_limits[1], ...,
    lon_limits[1]xlat_limits[0], ...].
    """

    hek_client = hek.HEKClient()
    start_time = aia_map.date - TimeDelta(1*u.hour)
    end_time = aia_map.date + TimeDelta(1*u.hour)

    responses = hek_client.search(a.Time(start_time, end_time),
                              a.hek.CH,
                              a.hek.FRM.Name == 'SPoCA')        # segmentation model: SPoCA
    
    geom_list = []
    for response in responses:
        if np.abs(response['hgc_y']) > 80.0:
            continue
        skycoord_obj = response['hpc_boundcc']
        coords = list(zip(skycoord_obj.Tx.value, skycoord_obj.Ty.value))
        g = Polygon(coords)
        if not g.is_valid:
            g = g.buffer(0)
        geom_list.append(g)

    # Merge all coronal hole areas from the responses
    merged = unary_union(geom_list)

    ny, nx = aia_map.data.shape
    y_idx, x_idx = np.indices((ny, nx))

    # Convert pixel coordinates to physical units
    xq = x_idx * u.pixel
    yq = y_idx * u.pixel

    # Convert pixel coordinates to world coordinates
    hpc_coords = aia_map.pixel_to_world(xq, yq)

    x_world = hpc_coords.Tx.value
    y_world = hpc_coords.Ty.value

    # Convert world coordinates to heliographic coordinates
    hgs_coords = hpc_coords.transform_to(frames.HeliographicStonyhurst)

    lon_deg = hgs_coords.lon.to(u.deg).value
    lat_deg = hgs_coords.lat.to(u.deg).value

    ch_mask = shapely.contains_xy(merged, x_world, y_world) # mask of coronal hole area

    A_CH = []

    for lon in lon_limits:
        central_lon_mask = (np.abs(lon_deg) <= lon)  # mask of longitudinal slice

        for lat_limit in lat_limits:
            central_lat_mask = np.abs(lat_deg) <= lat_limit  # mask of latitudinal slice
            central_mask = central_lon_mask & central_lat_mask # mask of central merdional slice

            inside_slice = central_mask.sum()                    # count of meridional slice pixels
            inside_ch_in_slice = (ch_mask & central_mask).sum()  # count of overlap pixels

            A_CH.append(inside_ch_in_slice / inside_slice if inside_slice > 0 else 0.0)

    return aia_map.date, *A_CH


# P_CH parameter
def get_P_CH(aia_map, lon_limits=[10], lat_limits=[30, 60, 90]):
    """
    Read the given sunpy.map.Map variable,
    and calculate the sum of the reciprocals of all pixel values within each
    central meridional region (±lon, ±lat).

    Return: (aia_map.date, P_CH) where P_CH is a flat list ordered lon-major,
    lat-minor (same convention as get_A_CH).
    """

    P_CH = []

    ny, nx = aia_map.data.shape

    for lon in lon_limits:
        n_lon = int(4 * lon + 1)
        lon_vals = np.linspace(-lon, lon, n_lon) * u.deg

        for lat in lat_limits:
            n_lat = int(4 * lat + 1)
            lat_vals = np.linspace(-lat, lat, n_lat) * u.deg

            # upper_boundary: latitude +lat degree
            upper_boundary_hgs = SkyCoord(lon=lon_vals,
                                        lat=lat*u.deg,
                                        frame=frames.HeliographicStonyhurst,
                                        obstime=aia_map.date,
                                        observer='earth')

            # lower_boundary: latitude -lat degree
            lower_boundary_hgs = SkyCoord(lon=lon_vals,
                                        lat=-lat*u.deg,
                                        frame=frames.HeliographicStonyhurst,
                                        obstime=aia_map.date,
                                        observer='earth')

            # left_boundary: longitude -lon degree
            left_boundary_hgs = SkyCoord(lon=-lon*u.deg,
                                        lat=lat_vals,
                                        frame=frames.HeliographicStonyhurst,
                                        obstime=aia_map.date,
                                        observer='earth')

            # right_boundary: longitude +lon degree
            right_boundary_hgs = SkyCoord(lon=lon*u.deg,
                                        lat=lat_vals,
                                        frame=frames.HeliographicStonyhurst,
                                        obstime=aia_map.date,
                                        observer='earth')

            upper_boundary_hpc = upper_boundary_hgs.transform_to(aia_map.coordinate_frame)
            lower_boundary_hpc = lower_boundary_hgs.transform_to(aia_map.coordinate_frame)
            left_boundary_hpc = left_boundary_hgs.transform_to(aia_map.coordinate_frame)
            right_boundary_hpc = right_boundary_hgs.transform_to(aia_map.coordinate_frame)

            upper_pix = aia_map.world_to_pixel(upper_boundary_hpc)
            lower_pix = aia_map.world_to_pixel(lower_boundary_hpc)
            left_pix = aia_map.world_to_pixel(left_boundary_hpc)
            right_pix = aia_map.world_to_pixel(right_boundary_hpc)

            xs = np.concatenate([
                np.array(lower_pix.x),
                np.array(right_pix.x),
                np.array(upper_pix.x)[::-1],
                np.array(left_pix.x)[::-1],
            ])

            ys = np.concatenate([
                np.array(lower_pix.y),
                np.array(right_pix.y),
                np.array(upper_pix.y)[::-1],
                np.array(left_pix.y)[::-1],
            ])

            vertices = np.vstack((xs, ys)).T   # shape (N_vertices, 2)
            poly = Path(vertices)

            x0 = max(int(np.floor(xs.min())), 0)
            x1 = min(int(np.ceil (xs.max())) + 1, nx)
            y0 = max(int(np.floor(ys.min())), 0)
            y1 = min(int(np.ceil (ys.max())) + 1, ny)

            Xb, Yb = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            points = np.vstack((Xb.ravel(), Yb.ravel())).T
            mask_bb = poly.contains_points(points)

            data_bb = aia_map.data[y0:y1, x0:x1].ravel()
            valid = mask_bb & (data_bb != 0)

            b = data_bb[valid]

            # 5% of the highest and lowest values are excluded.
            lower = np.percentile(b, 5)
            upper = np.percentile(b, 95)
            mask = (b >= lower) & (b <= upper)
            b_filtered = b[mask]

            P_CH.append(np.sum(np.reciprocal(b_filtered)))

    return aia_map.date, *P_CH
