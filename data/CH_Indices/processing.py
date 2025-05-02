"""
To get a CH_Indices such as A_CH, P_CH, theta(latitude).

These are our function to calculate solar parameters:
1. compute_A_CH
2. compute_P_CH
3. compute_theta

"""

import numpy as np
from matplotlib.path import Path

import astropy.units as u                   # 단위 처라
from astropy.coordinates import SkyCoord    # 천체 좌표계
from astropy.time import TimeDelta          # 시간 처리

from sunpy.map import Map
from sunpy.coordinates import frames        # 천체 좌표계
from sunpy.net import attrs as a            # 검색 조건 정의
from sunpy.net import hek                   # solar event 검색

import shapely.vectorized as sv             # numpy 배열 처리
from shapely import wkt
from shapely.ops import unary_union


# A_CH parameter
def get_A_CH(fits_file, lon=7.5):
    """
    주어진 FITS 파일을 읽어 HEK 검색으로 CH 이벤트를 찾고, 
    내부 픽셀 수 및 ±7.5° slice 영역을 이용해 A_CH 값을 계산합니다.
    
    반환: (aia_map.date, A_CH)
    """
    try:
        aia_map = Map(fits_file)
    except Exception as e:
        print(f"failed to open file: {fits_file} -> {e}")
        return None, None

    hek_client = hek.HEKClient()
    start_time = aia_map.date - TimeDelta(2*u.hour)
    end_time = aia_map.date + TimeDelta(2*u.hour)

    responses = hek_client.search(a.Time(start_time, end_time),
                              a.hek.CH,
                              a.hek.FRM.Name == 'SPoCA')        # segmentation model: SPoCA
    
    geom_list = []
    for response in responses:
        if np.abs(response['hgc_y']) > 80.0:
            continue
        g = wkt.loads(response['hpc_boundcc'])
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
    central_mask = (np.abs(lon_deg) <= lon)         # mask of central merdional slice
    ch_mask = sv.contains(merged, x_world, y_world) # mask of coronal hole area

    inside_slice = central_mask                     # mask of central merdional slice
    inside_ch_in_slice = ch_mask & inside_slice     # mask of coronal hole area & central merdional slice

    n_slice = inside_slice.sum()                    # count of meridional slice pixels
    n_ch_in_slice = inside_ch_in_slice.sum()        # count of overlap pixels

    A_CH = n_ch_in_slice / n_slice

    return aia_map.date, A_CH


# P_CH parameter
def get_P_CH(fits_file, lon=10, lat=30):
    """
    주어진 FITS 파일을 읽어 selected region 내의 모든 pixel values의 역수의 합을 계산합니다.
    
    반환: (aia_map.date, P_CH)
    """
    try:
        aia_map = Map(fits_file)
    except Exception as e:
        print(f"failed to open file: {fits_file} -> {e}")
        return None, None
    
    n_lon = int(4 * lon + 1)
    n_lat = int(4 * lat + 1)
    
    lon_vals = np.linspace(-lon, lon, n_lon) * u.deg
    lat_vals = np.linspace(-lat, lat, n_lat) * u.deg

    # upper_boundary: latitude +30 degree
    upper_boundary_hgs = SkyCoord(lon=lon_vals,
                                lat=lat*u.deg,
                                frame=frames.HeliographicStonyhurst,
                                obstime=aia_map.date,
                                observer='earth')

    # lower_boundary: latitude -30 degree
    lower_boundary_hgs = SkyCoord(lon=lon_vals,
                                lat=-lat*u.deg,
                                frame=frames.HeliographicStonyhurst,
                                obstime=aia_map.date,
                                observer='earth')

    # left_boundary: longitude -7.5 degree
    left_boundary_hgs = SkyCoord(lon=-lon*u.deg,
                                lat=lat_vals,
                                frame=frames.HeliographicStonyhurst,
                                obstime=aia_map.date,
                                observer='earth')

    # right_boundary: longitude +7.5 degree
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
        np.array(upper_pix.x),
        np.array(right_pix.x),
        np.array(lower_pix.x)[::-1],
        np.array(left_pix.x)[::-1]
    ])
    ys = np.concatenate([
        np.array(upper_pix.y),
        np.array(right_pix.y),
        np.array(lower_pix.y)[::-1],
        np.array(left_pix.y)[::-1]
    ])

    vertices = np.vstack((xs, ys)).T   # shape (N_vertices, 2)
    poly = Path(vertices)

    ny, nx = aia_map.data.shape

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
    P_CH = np.sum(np.reciprocal(b))

    return aia_map.date, P_CH


# theta parameter
def get_theta(fits_file):
    """
    
    """
    return