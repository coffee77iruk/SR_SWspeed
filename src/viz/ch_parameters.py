"""
Visualization helpers for coronal hole (CH) parameters (A_CH, P_CH).

Used to reproduce paper Figure 1: a 3x3 panel showing the SPoCA-segmented
CH boundaries and central meridional slice regions used to compute A_CH
(193 A) and P_CH (193/211 A) at latitude bands +-30/60/90.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta

from sunpy.coordinates import frames
from sunpy.net import attrs as a
from sunpy.net import hek
import sunpy.map

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.ndimage import binary_dilation

from data.ch.calibration.convert_to_level1_5 import convert_to_level1_5


_WAVE_RE = re.compile(r"\.(\d{3})\.image_lev1")


def load_and_calibrate(fits_file):
    """Load a level-1 AIA FITS file and calibrate it to level 1.5."""
    m = _WAVE_RE.search(os.path.basename(fits_file))
    if not m:
        raise ValueError(f"Could not parse wave band from filename: {fits_file}")
    wave = m.group(1)
    m0 = sunpy.map.Map(fits_file)
    m1_5 = convert_to_level1_5(m0)
    return wave, m1_5


def get_spoca_ch_union(aia_map, hours=1):
    """Query HEK for SPoCA CH boundaries near aia_map.date and merge them."""
    hek_client = hek.HEKClient()
    start_time = aia_map.date - TimeDelta(hours * u.hour)
    end_time = aia_map.date + TimeDelta(hours * u.hour)

    responses = hek_client.search(
        a.Time(start_time, end_time),
        a.hek.CH,
        a.hek.FRM.Name == 'SPoCA'
    )

    geom_list = []
    for r in responses:
        if np.abs(r['hgc_y']) > 80.0:
            continue
        skycoord_obj = r['hpc_boundcc']
        coords = list(zip(skycoord_obj.Tx.value, skycoord_obj.Ty.value))
        g = Polygon(coords)
        if not g.is_valid:
            g = g.buffer(0)
        geom_list.append(g)

    if len(geom_list) == 0:
        return None
    return unary_union(geom_list)


def make_hpc_hgs_grids(aia_map):
    """Return (x_world, y_world, lon_deg, lat_deg) pixel grids for aia_map."""
    ny, nx = aia_map.data.shape
    y_idx, x_idx = np.indices((ny, nx))
    xq = x_idx * u.pixel
    yq = y_idx * u.pixel

    hpc = aia_map.pixel_to_world(xq, yq)
    x_world = hpc.Tx.value
    y_world = hpc.Ty.value

    hgs = hpc.transform_to(frames.HeliographicStonyhurst)
    lon_deg = hgs.lon.to(u.deg).value
    lat_deg = hgs.lat.to(u.deg).value
    return x_world, y_world, lon_deg, lat_deg


def compute_A_CH(aia_map, merged_poly, lat_limits=(30.0, 60.0, 90.0), lon_limit=7.5):
    """
    Compute A_CH at each latitude band, along with the per-band slice mask
    and the full-disk CH mask (for plotting overlays).
    """
    x_world, y_world, lon_deg, lat_deg = make_hpc_hgs_grids(aia_map)

    if merged_poly is None:
        A = {L: np.nan for L in lat_limits}
        masks = {L: np.zeros_like(aia_map.data, dtype=bool) for L in lat_limits}
        ch_mask = np.zeros_like(aia_map.data, dtype=bool)
        return A, masks, ch_mask

    ch_mask = shapely.contains_xy(merged_poly, x_world, y_world)

    central_lon_mask = (np.abs(lon_deg) <= lon_limit)
    slice_masks = {L: (central_lon_mask & (np.abs(lat_deg) <= L)) for L in lat_limits}

    def frac(overlap_mask, slice_mask):
        denom = slice_mask.sum()
        return overlap_mask.sum() / denom if denom else np.nan

    A = {L: frac(ch_mask & slice_masks[L], slice_masks[L]) for L in lat_limits}
    return A, slice_masks, ch_mask


def compute_P_metric(aia_map, lat_limits=(30, 60, 90), lon_limit=10.0, clip_percent=(5, 95)):
    """
    Compute P_CH at each latitude band, along with the pixel-space boundary
    polygon (xs, ys) of each band's central meridional window (for plotting).
    """
    xs_list, ys_list, P_vals = [], [], []

    n_lon = int(4 * lon_limit + 1)
    lon_vals = np.linspace(-lon_limit, lon_limit, n_lon) * u.deg

    for lat in lat_limits:
        n_lat = int(4 * lat + 1)
        lat_vals = np.linspace(-lat, lat, n_lat) * u.deg

        upper_hgs = SkyCoord(lon=lon_vals, lat=lat * u.deg,
                             frame=frames.HeliographicStonyhurst,
                             obstime=aia_map.date, observer='earth')
        lower_hgs = SkyCoord(lon=lon_vals, lat=-lat * u.deg,
                             frame=frames.HeliographicStonyhurst,
                             obstime=aia_map.date, observer='earth')
        left_hgs = SkyCoord(lon=-lon_limit * u.deg, lat=lat_vals,
                            frame=frames.HeliographicStonyhurst,
                            obstime=aia_map.date, observer='earth')
        right_hgs = SkyCoord(lon=lon_limit * u.deg, lat=lat_vals,
                             frame=frames.HeliographicStonyhurst,
                             obstime=aia_map.date, observer='earth')

        upper_hpc = upper_hgs.transform_to(aia_map.coordinate_frame)
        lower_hpc = lower_hgs.transform_to(aia_map.coordinate_frame)
        left_hpc = left_hgs.transform_to(aia_map.coordinate_frame)
        right_hpc = right_hgs.transform_to(aia_map.coordinate_frame)

        upper_pix = aia_map.world_to_pixel(upper_hpc)
        lower_pix = aia_map.world_to_pixel(lower_hpc)
        left_pix = aia_map.world_to_pixel(left_hpc)
        right_pix = aia_map.world_to_pixel(right_hpc)

        xs = np.concatenate([np.array(lower_pix.x),
                             np.array(right_pix.x),
                             np.array(upper_pix.x)[::-1],
                             np.array(left_pix.x)[::-1]])
        ys = np.concatenate([np.array(lower_pix.y),
                             np.array(right_pix.y),
                             np.array(upper_pix.y)[::-1],
                             np.array(left_pix.y)[::-1]])

        vertices = np.vstack((xs, ys)).T
        poly = Path(vertices)

        ny, nx = aia_map.data.shape
        x0 = max(int(np.floor(xs.min())), 0)
        x1 = min(int(np.ceil(xs.max())) + 1, nx)
        y0 = max(int(np.floor(ys.min())), 0)
        y1 = min(int(np.ceil(ys.max())) + 1, ny)

        Xb, Yb = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        points = np.vstack((Xb.ravel(), Yb.ravel())).T
        mask_bb = poly.contains_points(points)

        data_bb = aia_map.data[y0:y1, x0:x1].ravel()
        valid = mask_bb & (data_bb != 0)
        b = data_bb[valid]

        if b.size == 0:
            P_vals.append(np.nan)
        else:
            lo = np.percentile(b, clip_percent[0])
            hi = np.percentile(b, clip_percent[1])
            b_f = b[(b >= lo) & (b <= hi)]
            P_vals.append(np.sum(1.0 / b_f) if b_f.size else np.nan)

        xs_list.append(xs)
        ys_list.append(ys)

    return P_vals, xs_list, ys_list


def map_extent_arcsec(aia_map):
    """Return [x0, x1, y0, y1] extent of aia_map in arcsec, for imshow(extent=...)."""
    ny, nx = aia_map.data.shape
    bl = aia_map.pixel_to_world(0 * u.pixel, 0 * u.pixel)
    tr = aia_map.pixel_to_world((nx - 1) * u.pixel, (ny - 1) * u.pixel)
    return [
        bl.Tx.to_value(u.arcsec), tr.Tx.to_value(u.arcsec),
        bl.Ty.to_value(u.arcsec), tr.Ty.to_value(u.arcsec),
    ]


def get_rgba_and_extent(aia_map, wave):
    """Return (rgba_image, extent) for the AIA colormap of the given wave."""
    if wave == '193':
        clipped = np.log10(aia_map.data.clip(40, 1e12))
    elif wave == '211':
        clipped = np.log10(aia_map.data.clip(10, 1e12))
    else:
        clipped = np.log10(aia_map.data.clip(1, 1e12))

    norm = colors.Normalize(vmin=np.nanmin(clipped), vmax=np.nanmax(clipped))
    cmap = plt.get_cmap(f'sdoaia{wave}')
    rgba = cmap(norm(clipped))
    extent = map_extent_arcsec(aia_map)
    return rgba, extent


def draw_ch_contour(ax, aia_map, poly, extent, color='cyan', linewidths=1.5):
    """Outline the SPoCA CH union polygon (from get_spoca_ch_union) on an
    already-imshow'd AIA panel. No-op if poly is None (no CH detected)."""
    if poly is None:
        return
    ny, nx = aia_map.data.shape
    y_idx, x_idx = np.indices((ny, nx))
    hpc = aia_map.pixel_to_world(x_idx * u.pixel, y_idx * u.pixel)
    mask = shapely.contains_xy(poly, hpc.Tx.value, hpc.Ty.value)
    xmin, xmax, ymin, ymax = extent
    ax.contour(
        np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny),
        mask.astype(int), levels=[0.5], colors=color, linewidths=linewidths, zorder=3,
    )


def set_hpc_axes(ax, extent, show_xlabel=True, show_ylabel=True,
                 label_fs=14, tick_fs=12, grid=True):
    """Apply consistent helioprojective-coordinate axis styling to ax."""
    ticks = [-1000, -500, 0, 500, 1000]

    ax.set_xlabel("Helioprojective Longitude (arcsec)" if show_xlabel else "", fontsize=label_fs)
    ax.set_ylabel("Helioprojective Latitude (arcsec)" if show_ylabel else "", fontsize=label_fs)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', labelsize=tick_fs)

    if grid:
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='white')


def plot_A_panel(ax, rgba_data, extent, ch_mask, slice_mask, A_val, L, title_fs=18):
    """Plot one A_CH panel: AIA image with CH mask + meridional slice overlay."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = ch_mask.shape

    emph = np.copy(rgba_data)
    emph[..., 3] = np.where(slice_mask, 1.0, 0.80)
    ax.imshow(emph, origin='lower', extent=extent)

    edge = binary_dilation(ch_mask) & ~ch_mask
    inside = ch_mask & slice_mask

    overlay = np.zeros_like(rgba_data)
    overlay[inside, 0:3] = 0.0
    overlay[inside, 3] = 0.60
    overlay[edge, 0:3] = 0.0
    overlay[edge, 3] = 0.85
    ax.imshow(overlay, origin='lower', extent=extent)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    ax.contour(x, y, slice_mask.astype(int), levels=[0.5],
               colors='snow', linewidths=1.5, linestyles='solid', zorder=3)
    ax.contour(x, y, ch_mask.astype(int), levels=[0.5],
               colors='cyan', linewidths=1.5, linestyles='solid', zorder=2)

    if np.isfinite(A_val):
        ax.set_title(rf"$A_{{CH{L}}}={A_val:.3f}$", fontsize=title_fs)
    else:
        ax.set_title(rf"$A_{{CH{L}}}=\mathrm{{NaN}}$", fontsize=title_fs)


def plot_P_panel(ax, rgba_data, extent, xs_pix, ys_pix, aia_map, P_val, L, wave, title_fs=18):
    """Plot one P_CH panel: AIA image with the meridional window boundary overlay."""
    ax.imshow(rgba_data, origin='lower', extent=extent)

    xq = xs_pix * u.pixel
    yq = ys_pix * u.pixel
    hpc = aia_map.pixel_to_world(xq, yq)
    xs = hpc.Tx.to_value(u.arcsec)
    ys = hpc.Ty.to_value(u.arcsec)

    ax.plot(xs, ys, color='white', linewidth=1.4)

    if np.isfinite(P_val):
        ax.set_title(rf"$P_{{CH{{{L}}},{{{wave}}}\,\mathrm{{Å}}}}={P_val:.1f}$", fontsize=title_fs)
    else:
        ax.set_title(rf"$P_{{CH{{{L}}},{{{wave}}}\,\mathrm{{Å}}}}=\mathrm{{NaN}}$", fontsize=title_fs)
