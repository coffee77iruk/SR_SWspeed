# a_ch_processor.py
import os
import glob
from datetime import timedelta

import numpy as np
import astropy.units as u                   # 단위 처라
from astropy.coordinates import SkyCoord    # 천체 좌표계
from astropy.time import TimeDelta          # 시간 처리

import sunpy
from sunpy.coordinates import frames        # 천체 좌표계
from sunpy.net import attrs as a            # 검색 조건 정의
from sunpy.net import hek                   # solar event 검색
from sunpy.physics.differential_rotation import solar_rotate_coordinate # 차등회전 보정
from sunpy.time import parse_time           # 시간 문자열 처리
from shapely.geometry import Point, Polygon # 다각형 처리
import shapely.vectorized as sv             # 다각형 벡터화 처리

from tqdm import tqdm
import warnings
from sunpy.util.exceptions import SunpyUserWarning
warnings.filterwarnings("ignore", category=SunpyUserWarning)

_fits_files_cache = {}

def get_fits_files(search_dir):
    """
    주어진 디렉토리(search_dir)에서 재귀적으로 모든 *.fits 파일을 glob으로 검색합니다.
    검색 결과는 캐시되어 이후 동일한 디렉토리 검색시 재사용됩니다.
    """
    if search_dir in _fits_files_cache:
        return _fits_files_cache[search_dir]
    else:
        pattern = os.path.join(search_dir, "**", "*.fits")
        files = glob.glob(pattern, recursive=True)
        _fits_files_cache[search_dir] = files
        return files

def extract_datetime_from_filename(fits_filepath):
    """
    Example for FITS filename:
      aia.lev1_euv_12s.2012-01-01T000009Z.193.image_lev1.fits
    에서 '2012-01-01T000009Z' 부분을 찾아 파싱하여 관측 시간을 반환합니다.
    """
    basename = os.path.basename(fits_filepath)
    parts = basename.split('.')
    for part in parts:
        if 'T' in part and part.endswith('Z'):
            timestr = part.rstrip('Z')
            try:
                date_part, time_part = timestr.split("T")
                if ":" not in time_part and len(time_part) == 6:
                    timestr_formatted = f"{date_part} {time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    # "000008" -> "00:00:08"
                    # timestr_formatted: 2012-01-01 00:00:02
                else:
                    timestr_formatted = timestr
                obs_time = parse_time(timestr_formatted)
                return obs_time
            except Exception as e:
                print(f"시간 파싱 실패: {fits_filepath} -> {e}")
                return None
    return None
# extract_datetime_from_filename(fits_filepath): "2012-01-01 00:00:02"

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
    start_time = aia_map.date - TimeDelta(2 * u.hour)
    end_time = aia_map.date + TimeDelta(2 * u.hour)

    responses = hek_client.search(a.Time(start_time, end_time),
                                  a.hek.CH,
                                  a.hek.FRM.Name == 'SPoCA')

    area = 0.0
    response_index = None
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
    rotated_ch_boundary = solar_rotate_coordinate(ch_boundary, time=aia_map.date)
    
    # generate a polygon from the rotated CH boundary coordinates
    try:

        ch_polygon = Polygon(
            tx = rotated_ch_boundary.Tx.value
            ty = rotated_ch_boundary.Ty.value
            valid = np.isfinite(tx) & np.isfinite(ty)
            if np.count_nonzero(valid) < 3:
                raise ValueError("유효한 경계 좌표가 부족합니다.")
            zip(tx[valid], ty[valid])
            #zip(rotated_ch_boundary.Tx.value, rotated_ch_boundary.Ty.value)
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
    roi_inside = sv.contains(ch_polygon, roi_pix_x, roi_pix_y)

    overlap_mask = np.zeros_like(mask_lon, dtype=bool)
    overlap_mask[roi_mask] = roi_inside             # inside CH and within ±7.5° slice

    overlap_count = np.count_nonzero(overlap_mask)  # count of overlap pixels
    slice_pixel_count = np.count_nonzero(mask_lon)  # count of meridional slice pixels

    A_CH = overlap_count / slice_pixel_count if slice_pixel_count else float('nan')

    return aia_map.date, A_CH

def process_a_ch(channel, start_dt, end_dt, cadence_hours, base_dir, time_tolerance_sec=600):
    """
    지정한 채널 폴더에서 시작 시간부터 종료 시간까지 cadence_hours 간격으로 
    FITS 파일을 검색하여 가장 가까운 파일을 선택하고 A_CH 값을 계산한 후
    결과 리스트(각 행: "YYYY-MM-DD HH A_CH_value")를 반환합니다.
    
    입력:
      - channel: EUV 파장대 (예: "193", "211")
      - start_dt, end_dt: datetime 객체
      - cadence_hours: 간격 (시간 단위)
      - base_dir: 부모 폴더 (예: r"E:\Research\SR\input\CH")
      - time_tolerance_sec: 지정 시간과 파일 관측시간의 허용 오차 (초)
    """
    results = []
    step = timedelta(hours=cadence_hours)   # cadence
    total_steps = int(((end_dt - start_dt).total_seconds() // step.total_seconds())) + 1
    
    # Get all years between start and end dates.
    years = set()
    current_time = start_dt
    while current_time <= end_dt:
        years.add(current_time.year)
        current_time += step
    # years: {2012, 2013, 2014, ...}

    # Precompute the FITS files for each year.
    precomputed_fits = {}
    for year in sorted(years):
        search_dir = os.path.join(base_dir, channel, str(year))
        if not os.path.isdir(search_dir):
            precomputed_fits[year] = None
            print(f"폴더가 존재하지 않습니다: {search_dir}")
        else:
            precomputed_fits[year] = get_fits_files(search_dir)
            if not precomputed_fits[year]:
                print(f"{search_dir}에서 FITS 파일이 발견되지 않았습니다.")
            else:
                print("fits_files를 precomputed_fits에 저장합니다.")
    # precomputed_fits[year]: {year: [fits_file1, fits_file2, ...]}

    # Iterate through the time range and process each step.
    current_time = start_dt
    for _ in tqdm(range(total_steps), desc="Processing FITS files", unit="step"): # 수정 -> year로
        year = current_time.year                # year of current_time
        fits_files = precomputed_fits.get(year) # fits_files for the current year

        if fits_files is None:
            results.append(f"{current_time.strftime('%Y-%m-%d %H')} NoFolder")
            current_time += step
            continue
        if not fits_files:
            results.append(f"{current_time.strftime('%Y-%m-%d %H')} NoFile")
            current_time += step
            continue


        best_file = None
        for f in fits_files:
            obs_time = extract_datetime_from_filename(f)
            if obs_time is None:
                continue
            obs_dt = obs_time.to_datetime()
            # "YYYY-MM-DD HH" 형식으로 시간 비교를 수행합니다.
            if obs_dt.strftime('%Y-%m-%d %H') == current_time.strftime('%Y-%m-%d %H'):
                best_file = f
                break  # 일치하는 파일이 있으면 바로 선택합니다.

        #best_file = None
        #best_diff = None
        # Iterate through the FITS files to find the closest one.
        #for f in fits_files:
        #    obs_time = extract_datetime_from_filename(f)    
        #    # f: 'E:\\Research\\SR\\input\\CH\\211\\2012\\aia.lev1_euv_12s.2012-01-01T000002Z.211.image_lev1.fits'
        #    # obs_time: '2012-01-01 00:00:02'
        #    if obs_time is None:
        #        continue
        #    obs_dt = obs_time.to_datetime()
        #    diff = abs((obs_dt - current_time).total_seconds())
        #    if diff <= time_tolerance_sec:
        #        if best_diff is None or diff < best_diff:
        #            best_diff = diff
        #            best_file = f

        if best_file:
            print(f"{current_time}에 해당하는 파일: {best_file}")
            obs_time, A_CH = compute_A_CH(best_file)
            if A_CH is not None:
                results.append(f"{obs_time.strftime('%Y-%m-%d %H')} {A_CH:.4f}")
            else:
                results.append(f"{current_time.strftime('%Y-%m-%d %H')} NaN")
        else:
            print(f"{current_time}에 해당하는 파일이 없습니다.")
            results.append(f"{current_time.strftime('%Y-%m-%d %H')} NoFile")

        current_time += step

    return results
