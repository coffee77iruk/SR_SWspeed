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

import sunpy
from sunpy.coordinates import frames        # 천체 좌표계
from sunpy.net import attrs as a            # 검색 조건 정의
from sunpy.net import hek                   # solar event 검색
from sunpy.physics.differential_rotation import solar_rotate_coordinate # 차등회전 보정
from sunpy.time import parse_time           # 시간 문자열 처리
from shapely.geometry import Polygon        # 다각형 처리
from shapely import contains_xy             # 다각형 벡터화 처리


# A_CH parameter
def compute_A_CH(fits_file):
    """
    주어진 FITS 파일을 읽어 HEK 검색으로 CH 이벤트를 찾고, 
    내부 픽셀 수 및 ±7.5° slice 영역을 이용해 A_CH 값을 계산합니다.
    
    반환: (aia_map.date, A_CH)
    """
    try:
        aia_map = sunpy.map.Map(fits_file)
    except Exception as e:
        print(f"failed to open file: {fits_file} -> {e}")
        return None, None

    hek_client = hek.HEKClient()
    start_time = aia_map.date - TimeDelta(2*u.hour)
    end_time = aia_map.date + TimeDelta(2*u.hour)

    responses = hek_client.search(a.Time(start_time, end_time),
                                  a.hek.CH,
                                  a.hek.FRM.Name == 'SPoCA')    # segmentation model: SPoCA

    area = 0.0
    for i, response in enumerate(responses):
        if response['area_atdiskcenter'] > area and np.abs(response['hgc_y']) < 80.0:
            area = response['area_atdiskcenter']
            response_index = i

    if response_index is None:
        print(f"No CH even in fits file: {fits_file}")
        return aia_map.date, None

    ch = responses[response_index]
    
    # Process the coronal hole boundary string to extract coordinates
    p1 = ch["hpc_boundcc"][9:-2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]
    ch_date = parse_time(ch['event_starttime'])

    """"The coronal hole was detected at different time than the AIA image was taken so we need to rotate it to the map observation time."""
    ch_boundary = SkyCoord([(float(v[0]), float(v[1])) * u.arcsec for v in p3],
                           obstime=ch_date,                 # evebt start time
                           observer="earth",                # observer: Earth
                           frame=frames.Helioprojective)    # frame: Helioprojective
    rotated_ch_boundary = solar_rotate_coordinate(ch_boundary, 
                                                  time=aia_map.date)
    
    # generate a polygon from the rotated CH boundary coordinates
    try:
        ch_polygon = Polygon(
            zip(rotated_ch_boundary.Tx.value, rotated_ch_boundary.Ty.value)
        )

    except Exception as e:
        print(f"fail to generate polygon of CH boundary: {fits_file} -> {e}")
        return aia_map.date, None

    ny, nx = aia_map.data.shape
    yy, xx = np.indices((ny, nx))

    # Convert pixel coordinates to physical units
    xx_q = xx * u.pixel
    yy_q = yy * u.pixel

    # Convert pixel coordinates to world coordinates
    world_coords = aia_map.pixel_to_world(xx_q, yy_q)

    # Convert world coordinates to heliographic coordinates
    hgs_coords = world_coords.transform_to(frames.HeliographicStonyhurst)
    lon_deg = hgs_coords.lon.to(u.deg).value    # degree
    mask_lon = (np.abs(lon_deg) <= 7.5)         # if lon_deg <= ±7.5, mask_lon = True

    pix_x = world_coords.Tx.value
    pix_y = world_coords.Ty.value

    # restrict the mask to the area of the map
    roi_mask = mask_lon
    roi_pix_x = pix_x[roi_mask]
    roi_pix_y = pix_y[roi_mask]

    valid_points = np.isfinite(roi_pix_x) & np.isfinite(roi_pix_y)
    # roi_pix_x, roi_pix_y: only valid points in the map
    roi_pix_x = roi_pix_x[valid_points]
    roi_pix_y = roi_pix_y[valid_points]
    roi_inside = contains_xy(ch_polygon, roi_pix_x, roi_pix_y)

    overlap_mask = np.zeros_like(mask_lon, dtype=bool)
    overlap_mask[roi_mask] = roi_inside             # inside CH and within ±7.5° slice

    overlap_count = np.count_nonzero(overlap_mask)  # count of overlap pixels
    slice_pixel_count = np.count_nonzero(mask_lon)  # count of meridional slice pixels

    A_CH = overlap_count / slice_pixel_count if slice_pixel_count else float('nan')

    return aia_map.date, A_CH


# P_CH parameter
def compute_P_CH(fits_file, lon=7.5, lat=30):
    """
    주어진 FITS 파일을 읽어 selected region 내의 모든 pixel values의 역수의 합을 계산합니다.
    
    반환: (aia_map.date, P_CH)
    """
    try:
        aia_map = sunpy.map.Map(fits_file)
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
def compute_theta(fits_file):
    return