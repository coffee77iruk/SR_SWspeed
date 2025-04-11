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
                    # "000008" -> "00:00:08"
                    timestr_formatted = f"{date_part}T{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                else:
                    timestr_formatted = timestr
                obs_time = parse_time(timestr_formatted)
                return obs_time
            except Exception as e:
                print(f"시간 파싱 실패: {fits_filepath} -> {e}")
                return None
    return None

def compute_A_CH(fits_file):
    """
    주어진 FITS 파일을 읽어 HEK 검색으로 CH 이벤트를 찾고, 
    내부 픽셀 수 및 ±7.5° slice 영역을 이용해 A_CH 값을 계산합니다.
    
    반환: (aia_map.date, A_CH)
    """
    try:
        aia_map = sunpy.map.Map(fits_file)
    except Exception as e:
        print(f"파일 열기 실패: {fits_file} -> {e}")
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
        print(f"CH 이벤트 없음: {fits_file}")
        return aia_map.date, None

    ch = responses[response_index]
    
    # CH 경계 문자열 처리 (원본 코드의 일부를 단순화)
    p1 = ch["hpc_boundcc"][9:-2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]

    ch_date = parse_time(ch['event_starttime'])
    ch_boundary = SkyCoord([(float(v[0]), float(v[1])) * u.arcsec for v in p3],
                           obstime=ch_date,
                           observer="earth",
                           frame=frames.Helioprojective)
    rotated_ch_boundary = solar_rotate_coordinate(ch_boundary, time=aia_map.date)
    
    # CH 경계 다각형 생성 (좌표 값들이 문자열로 들어가 있을 수 있으므로 변환)
    try:
        ch_polygon = Polygon(
            zip(rotated_ch_boundary.Tx.value, rotated_ch_boundary.Ty.value)
        )
    except Exception as e:
        print(f"CH 경계 다각형 생성 실패: {fits_file} -> {e}")
        return aia_map.date, None

    ny, nx = aia_map.data.shape
    yy, xx = np.indices((ny, nx))
    xx_q = xx * u.pixel
    yy_q = yy * u.pixel
    world_coords = aia_map.pixel_to_world(xx_q, yy_q)

    hgs_coords = world_coords.transform_to(frames.HeliographicStonyhurst)
    lon_deg = hgs_coords.lon.to(u.deg).value
    mask_lon = (np.abs(lon_deg) <= 7.5)

    pix_x = world_coords.Tx.value
    pix_y = world_coords.Ty.value
    pix_x_flat = pix_x.ravel()
    pix_y_flat = pix_y.ravel()
    points = [Point(x, y) for x, y in zip(pix_x_flat, pix_y_flat)]
    inside_ch_flat = [ch_polygon.contains(pt) for pt in points]
    inside_ch = np.array(inside_ch_flat).reshape(pix_x.shape)

    overlap_mask = inside_ch & mask_lon
    overlap_count = np.count_nonzero(overlap_mask)
    slice_pixel_count = np.count_nonzero(mask_lon)
    A_CH = overlap_count / slice_pixel_count if slice_pixel_count else float('nan')

    return aia_map.date, A_CH

def process_a_ch(channel, start_dt, end_dt, cadence_hours, base_dir, time_tolerance_sec=600):
    """
    지정한 채널 폴더에서 시작 시간부터 종료 시간까지 cadence_hours 간격으로 
    FITS 파일을 검색하여 가장 가까운 파일을 선택하고 A_CH 값을 계산한 후
    결과 리스트(각 행: "YYYY-MM-DD HH A_CH_value")를 반환합니다.
    
    입력:
      - channel: 채널 폴더 (예: "193", "211")
      - start_dt, end_dt: datetime 객체
      - cadence_hours: 간격 (시간 단위)
      - base_dir: 최상위 폴더 (예: r"E:\Research\SR\input\CH")
      - time_tolerance_sec: 지정 시간과 파일 관측시간의 허용 오차 (초)
    """
    results = []
    step = timedelta(hours=cadence_hours)   # cadence 단위
    total_steps = int(((end_dt - start_dt).total_seconds() // step.total_seconds())) + 1
    current_time = start_dt

    for _ in tqdm(range(total_steps), desc="Processing FITS files", unit="step"):
        year = current_time.year
        search_dir = os.path.join(base_dir, channel, str(year))
        if not os.path.isdir(search_dir):
            print(f"폴더가 존재하지 않습니다: {search_dir}")
            results.append(f"{current_time.strftime('%Y-%m-%d %H')} NoFolder")
            current_time += step
            continue

        # glob 검색 부분을 분리한 함수 호출
        fits_files = get_fits_files(search_dir)

        if not fits_files:
            print(f"{search_dir}에서 FITS 파일이 발견되지 않았습니다.")
            results.append(f"{current_time.strftime('%Y-%m-%d %H')} NoFile")
            current_time += step
            continue

        best_file = None
        best_diff = None
        for f in fits_files:
            obs_time = extract_datetime_from_filename(f)
            if obs_time is None:
                continue
            obs_dt = obs_time.to_datetime()
            diff = abs((obs_dt - current_time).total_seconds())
            if diff <= time_tolerance_sec:
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_file = f

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
