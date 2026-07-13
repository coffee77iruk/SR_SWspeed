"""
This code is for performance verification for symbolic regression model.

You can use filter_and_round() to set an acceptable time window for magnetic indices.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from datetime import datetime, timedelta
from astropy.time import Time
from tqdm import tqdm

import os

import requests
from bs4 import BeautifulSoup
from io import StringIO

# Importing Necessary Libraries for Performance Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

# Importing Necessary Libraries for Plotting Taylor Diagram
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa

import skill_metrics as sm
from matplotlib import rcParams
#from matplotlib.lines import Line2D

# Solar Wind Speed Models
import sys
# SW_MODELS_PATH 환경변수로 경로 설정 가능 (기본값: 로컬 연구 디렉터리)
_SW_MODELS_PATH = os.environ.get(
    "SW_MODELS_PATH",
    r"C:\Users\user\Research\SW_Speed_models_comparison\data"
)
sys.path.append(_SW_MODELS_PATH)
from omni import OMNI
from wsa_enlil import WSA_ENLIL
from eswf3_2 import ESWF3_2
from dl_model import DL_Model
from persistence import Persistence_27days

"""
Example of path
magnetic_path = "E:/research/SW_Speed_SR/data/input/mag_parameters/R21_5/PREDSOLARWIND/GONGZfield_line1R000.dat"
CH_path       = "E:/research/SW_Speed_SR/data/input/CH_parameters/CH_param_cad1.csv"
output_path   = "E:/research/SW_Speed_SR/data/output/omni2_2000-2024.lst"
"""

# =============================================================================
# Shared utility
# =============================================================================
def fetch_icme_events(year_start: int = 2010, year_end: int = 2024) -> list:
    """
    Fetch ICME event list from the Caltech ACE ICME catalog and return
    a list of [start_str, end_str] pairs in '%Y-%m-%dT%H:%M:%S' format.
    """
    url = "https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"
    response = requests.get(url)

    if response.status_code != 200:
        raise ConnectionError(f"ICME catalog request failed: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    icme_entire_table = pd.read_html(StringIO(str(table)))[0]

    icme_sub_table = icme_entire_table.iloc[:, [1, 2, 11, 12]].copy()
    icme_sub_table.columns = ['ICME_start', 'ICME_end', 'ICME_mean', 'ICME_max']
    icme_sub_table = icme_sub_table.dropna(subset=['ICME_start', 'ICME_end'])

    icme_sub_table['ICME_start'] = pd.to_datetime(
        icme_sub_table['ICME_start'], errors='coerce', format='%Y/%m/%d %H%M'
    )
    icme_sub_table['ICME_end'] = pd.to_datetime(
        icme_sub_table['ICME_end'], errors='coerce', format='%Y/%m/%d %H%M'
    )

    mask = (
        (icme_sub_table['ICME_start'].dt.year >= year_start) &
        (icme_sub_table['ICME_start'].dt.year <= year_end) &
        (icme_sub_table['ICME_end'].dt.year >= year_start) &
        (icme_sub_table['ICME_end'].dt.year <= year_end)
    )
    icme_df = icme_sub_table[mask].reset_index(drop=True)

    return [
        [row['ICME_start'].strftime('%Y-%m-%dT%H:%M:%S'),
         row['ICME_end'].strftime('%Y-%m-%dT%H:%M:%S')]
        for _, row in icme_df.iterrows()
    ]

# =============================================================================
# 1. SRIndicesProcessor - Index & Data Preprocessing for Symbolic Regression
# =============================================================================
class SRIndicesProcessor:
    """
    Class to process solar wind speed related indices:
    - Magnetic indices 
      1. expansion factor
      2. angular distance between footpoint and nearest coronal hole boundary
      3. squashing factor
    - Coronal hole indices:
      1. fractional coronal hole area
      2. coronal hole brightness
    - ICME events exclusion
    
    Main use: Preprocessing input features for symbolic regression models.

    """

    def __init__(self, magnetic_path, CH_path, output_path, bias=2440000.00):
        self.magnetic_path = magnetic_path
        self.CH_path = CH_path
        self.output_path = output_path
        self.bias = bias

    # jd2dt and filter_and_round methods are related to magnetic indices processing
    def jd2dt(self, jd_series):
        """Convert Julian date to datetime."""
        return Time(jd_series + self.bias, format="jd").to_datetime()

    def filter_and_round(self, ts: pd.Timestamp):
        m = ts.minute
        if 20 <= m <= 40:
            return None
        if m < 20:
            return ts.replace(minute=0, second=0, microsecond=0)
        nxt = ts + pd.Timedelta(hours=1)
        return nxt.replace(minute=0, second=0, microsecond=0)

    def get_omni_df(self):
        rows = []
        with open(self.output_path, "r") as f:
            for line in f:
                year, doy, hour, speed = line.strip().split()
                year, doy, hour, speed = int(year), int(doy), int(hour), float(speed)
                if (year >= 2010) and (hour in (0, 3, 6, 9, 12, 15, 18, 21)):
                    dt = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour)
                    dt = dt.strftime("%Y-%m-%dT%H:%M:%S")
                    rows.append({"datetime": dt, "speed": speed})
        omni_speed = pd.DataFrame(rows, columns=["datetime", "speed"])
        omni_speed['datetime'] = pd.to_datetime(omni_speed['datetime'])
        omni_speed.loc[omni_speed['speed'] > 1000, 'speed'] = np.nan  # Process outliers
        return omni_speed

    def set_magnetic_indices(self):
        magnetic_df = pd.read_csv(self.magnetic_path, comment='#', sep=r'\s+', engine='python')
        magnetic_df["departure_time"] = self.jd2dt(magnetic_df["parcel_depart_time"])
        magnetic_df["arrival_time"]   = self.jd2dt(magnetic_df["juldate"])
        magnetic_df['datetime'] = magnetic_df['arrival_time'].apply(self.filter_and_round)
        magnetic_df = magnetic_df.dropna(subset=['datetime'])

        valid_hours = list(range(0, 24, 3))
        mask = (
            magnetic_df['datetime'].dt.year.between(2010, 2024)
            & magnetic_df['datetime'].dt.hour.isin(valid_hours)
        )
        # magnetic indices columns
        magnetic_df = magnetic_df.loc[mask, [
            'datetime',
            'departure_time',
            'arrival_time',
            'expansion_factor',
            'coronal_hole_dist',
            'squashing_factor',
            'B_footpoint',
            'Vp'
        ]]

        # Average over the same time
        """ Note: I'm not sure if this is the right way to average the data. """
        magnetic_df = magnetic_df.groupby('datetime', as_index=False).mean()
        magnetic_df.loc[magnetic_df['expansion_factor'] > 5000, 'expansion_factor'] = np.nan  # Process outliers
        return magnetic_df
    
    def set_ch_indices(self):
        ch_df = pd.read_csv(self.CH_path)
        # Process outliers
        cols = ['P_CH30_193', 'P_CH90_193', 'P_CH30_211', 'P_CH90_211']
        for _ in range(5):
            for col in cols:
                mu = ch_df[col].mean()
                sigma = ch_df[col].std()
                mask = (ch_df[col] < mu - 3*sigma) | (ch_df[col] > mu + 3*sigma) | (ch_df[col] <= 0)
                if not mask.any():
                    break
                ch_df.loc[mask, col] = np.nan

        # CH indices columns
        CH_df = pd.DataFrame({
            'datetime': ch_df['datetime'],
            # Lagged features for CH_193
            'A_CH_193_lag4': ch_df['A_CH_193'].shift(96),
            'P_CH30_193_lag3p5':  ch_df['P_CH30_193'].shift(84),
            'P_CH30_193_lag4':    ch_df['P_CH30_193'].shift(96),
            'P_CH30_193_lag4p5':  ch_df['P_CH30_193'].shift(108),
            # Lagged features for CH_211
            'A_CH_211_lag4':    ch_df['A_CH_211'].shift(96),
            'P_CH30_211_lag4':    ch_df['P_CH30_211'].shift(96),
            'P_CH30_211_lag4p5':  ch_df['P_CH30_211'].shift(108),
        })
        CH_df['datetime'] = pd.to_datetime(CH_df['datetime'])
        return CH_df

    def get_merged_df(self):
        merged_df = self.get_omni_df().merge(
            self.set_magnetic_indices()[[
                'datetime', 'expansion_factor', 'coronal_hole_dist', 'squashing_factor', 'B_footpoint'
                ]],
            on='datetime',
            how='left'
        )
        merged_df = merged_df.merge(
            self.set_ch_indices(),
            on='datetime',
            how='left'
        )
        return merged_df

    def get_icme_events(self) -> list:
        return fetch_icme_events()


# =============================================================================
# 2. Statistical_Verification - Metrics Calculation & Evaluation
# =============================================================================
class Statistical_verification:
    """
    Calculates statistical metrics (MAE, RMSE, CC) for prediction models.
    - Supports ICME exclusion
    - Accepts raw + model data, returns performance summary
    """
    def __init__(self, model_list=[]):
        self.model_list = model_list  # e.g., ['wsa_speeds', 'eswf3_2', 'model_speeds', 'persistence27']
        self.CR_DICT = {
            "2012": [2128, 2129, 2130, 2131, 2132],
            "2013": [2142, 2143, 2144, 2145],
            "2014": [2155, 2156, 2157, 2158],
            "2015": [2168, 2169, 2170, 2171, 2172],
            "2016": [2182, 2183, 2184, 2185],
            "2017": [2195, 2196, 2197, 2198, 2199],
            "2018": [2209, 2210, 2211, 2212],
            "2019": [2222, 2223, 2224, 2225],
            "2020": [2235, 2236, 2237, 2238, 2239]
        }

    def get_icme_events(self) -> list:
        return fetch_icme_events()

    # Importing solar wind speed models data
    def build_sw_models_df(self, TW_years):
        """ OMNI, WSA-ENLIL, ESWF3.2, DL model, and 27-days persistence data are all combined 
        and returned to one DataFrame.
        """
        MODEL_MAP = {
            'wsa_speeds'   :  (WSA_ENLIL,          'wsa_speeds'),
            'eswf3_2'      :  (ESWF3_2,            'eswf3_2'),
            'model_speeds' :  (DL_Model,           'model_speeds'),
            'persistence27':  (Persistence_27days, 'persistence27'),
        }
        frames = []
        pbar = tqdm(TW_years, desc="Processing years")
        for year in pbar:
            pbar.set_postfix(current_year=year)
            CR_LIST = self.CR_DICT.get(str(year))
            if str(year) not in self.CR_DICT:
                raise ValueError(f"Year {year} is not available in the CR_DICT. Available years: {list(self.CR_DICT.keys())}")

            for cr in CR_LIST:
                dates, omni  = OMNI(cr)
                df = pd.DataFrame({
                    'datetime': pd.to_datetime(dates, format='%Y-%m-%d %H:%M'),
                    'omni_speeds': omni,
                })
                for model in self.model_list:
                    if model in MODEL_MAP:
                        func, col = MODEL_MAP[model]
                        _, vals = func(cr)
                        df[col] = vals
                frames.append(df)
        
        models_df = pd.concat(frames, ignore_index=True)
        models_df = models_df.dropna(subset=['datetime'])
        return models_df
        
    # Calculate performance metrics
    def get_df_to_calculate(self, df, TW_years=[2015, 2016, 2017, 2018], TW_months=[10, 11, 12]):
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        mask = pd.Series(False, index=df.index)
        for start_str, end_str in self.get_icme_events():
            start_dt = pd.to_datetime(start_str)
            end_dt   = pd.to_datetime(end_str)
            mask |= df['datetime'].between(start_dt, end_dt)

        cols = df.columns.drop('datetime')
        df.loc[mask, cols] = np.nan
        df = df[df['datetime'].dt.year.isin(TW_years)].reset_index(drop=True)
        df = df[df['datetime'].dt.month.isin(TW_months)].reset_index(drop=True)
        
        """ Merge with solar wind models """
        model_df = self.build_sw_models_df(TW_years)
        cols_to_add = [col for col in model_df.columns if col != 'datetime']
        df = df.merge(
            model_df[['datetime'] + cols_to_add],
            on='datetime',
            how='left'
        )
        return df

    def calculate_metrics(self, df, model_cols, target_col):
        save_models_performance = []
        for model_col in model_cols:
            save_metrics = {}
            if model_col not in df.columns:
                print(f"Column {model_col} not found in DataFrame.")
                continue
            # Delete the rows with NaN values
            temp_df = df[[target_col, model_col]].dropna()
            
            y_true = temp_df[target_col].values
            y_pred = temp_df[model_col].values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            cc = pearsonr(y_true, y_pred)[0]

            save_metrics[model_col] = {
                'MAE': mae,
                'RMSE': rmse,
                'CC': cc
            }
            save_models_performance.append(save_metrics)

        return save_models_performance

# =============================================================================
# 3. TaylorDiagram - Visualization Class
# =============================================================================
class TaylorDiagram(object):
    """
    Taylor diagram plotting utility for SR model comparison.
    """
    def __init__(self, STD ,fig=None, rect=111, label='_'):
        self.STD = STD
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
        tlocs = np.arccos(rlocs) # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs) # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.6 * self.STD
        gh = fa.GridHelperCurveLinear(tr,extremes=(0,(np.pi/2),self.smin,self.smax),grid_locator1=gl1,tick_formatter1=tf1,)
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)
        # Angle axis
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].label.set_text("Correlation coefficient")
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        # X axis
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_text("Standard deviation [km/s]")
        ax.axis['left'].toggle(ticklabels=True, label=True)
        ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
        ax.axis['left'].label.set_axis_direction('bottom')
        # Y axis
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].label.set_text("Standard deviation [km/s]")
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        ax.axis['right'].label.set_axis_direction('top')
        # Useless
        ax.axis['bottom'].set_visible(False)
        # Contours along standard deviations
        ax.grid()
        self._ax = ax # Graphical axes
        self.ax = ax.get_aux_axes(tr) # Polar coordinates
        # Add reference point and STD contour
        l, = self.ax.plot([0], self.STD, 'k*', ls='', ms=15, label=label)
        t = np.linspace(0, (np.pi / 2.0))
        r = np.zeros_like(t) + self.STD
        self.ax.plot(t, r, 'k--', label='_')
        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]
    
    def add_sample(self, STD, r, *args, **kwargs):
        l, = self.ax.plot(np.arccos(r), STD, *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)
        return l

    def add_contours(self,levels=5,**kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
        RMSE=np.sqrt(np.power(self.STD, 2) + np.power(rs, 2) - (2.0 * self.STD * rs  *np.cos(ts)))
        contours = self.ax.contour(ts, rs, RMSE, levels, **kwargs)
        return contours

# =============================================================================
# 4. create_taylor_diagram - Wrapper Function
# =============================================================================
def create_taylor_diagram(SR_df_list, sr_datasets=None, save_figure=False,
                          save_figure_name="Taylor Diagram of SR model",
                          save_dir="."):
    omni, wsa_enlil, persist27 = [], [], []
    for solar_phases in SR_df_list:
        solar_phases = solar_phases.dropna(subset=['speed', 'wsa_speeds', 'persistence27'])
        omni.append(solar_phases['speed'])
        wsa_enlil.append(solar_phases['wsa_speeds'])
        persist27.append(solar_phases['persistence27'])

    des_omni, des_wsa_enlil, des_persist27 = np.array(omni[0]), np.array(wsa_enlil[0]), np.array(persist27[0])
    min_omni, min_wsa_enlil, min_persist27 = np.array(omni[1]), np.array(wsa_enlil[1]), np.array(persist27[1])

    model_taylor_stats = sm.taylor_statistics(des_omni, des_omni)

    wsa_descending_taylor_stats       = sm.taylor_statistics(des_wsa_enlil, des_omni)
    wsa_minimum_taylor_stats          = sm.taylor_statistics(min_wsa_enlil, min_omni)
    persist27_descending_taylor_stats = sm.taylor_statistics(des_persist27, des_omni)
    persist27_minimum_taylor_stats    = sm.taylor_statistics(min_persist27, min_omni)

    obsSTD = model_taylor_stats['sdev'][0]

    datasets = [
        {"stats": wsa_descending_taylor_stats,       "color": "deepskyblue",  "label": "WSA-Enlil: Descending",       "marker": "s"},
        {"stats": wsa_minimum_taylor_stats,          "color": "deepskyblue",  "label": "WSA-Enlil: Minimum",          "marker": "^"},
        {"stats": persist27_descending_taylor_stats, "color": "orange",       "label": "27-days Persistence: Descending", "marker": "s"},
        {"stats": persist27_minimum_taylor_stats,    "color": "orange",       "label": "27-days Persistence: Minimum",    "marker": "^"},
    ]

    if sr_datasets:
        for data in sr_datasets:
            datasets.append(data)

    # Create the Taylor Diagram
    fig = plt.figure(figsize=(10, 8))
    dia = TaylorDiagram(obsSTD, fig=fig, rect=111, label='OMNI')

    # Add Contours
    plt.clabel(dia.add_contours(colors='#808080'), inline=1, fontsize=10)

    # Add each data points
    for data in datasets:
        stats = data["stats"]
        color = data["color"]
        label = data["label"]
        marker = data["marker"]
        fill = data.get("fill", False)

        dia.add_sample(
            stats['sdev'][1],  # standard deviation
            stats['ccoef'][1], # correlation coefficient
            label=label,
            marker=marker,
            mec=color,
            mfc=color if fill else 'none',  # fill color if entire period
            mew=2.0,  # width of the marker edge
            ms=12,    # size of the marker
        )

    # Add legend
    spl = [p.get_label() for p in dia.samplePoints]
    legend = fig.legend(
        dia.samplePoints,
        spl,
        numpoints=1,
        loc='upper right',    # location of the legend
        prop=dict(size=9.2),  # text size of the legend
        frameon=True,
        handlelength=2,  # length of the legend marker
        handleheight=2,  # hight of the legend marker
    )

    # remove the frame of the legend
    for handle in legend.legend_handles:
        handle.set_linewidth(0)

    plt.title("Taylor Diagram", fontsize=20, pad=40)

    # control the plot appearance
    plt.rcParams["figure.figsize"] = [10.0, 8.0]    # set figure size more wide
    plt.rcParams['lines.linewidth'] = 2             # set line width more thick
    plt.rcParams.update({'font.size': 15})          # set fontsize more large

    if save_figure:
        save_path = os.path.join(save_dir, f"{save_figure_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# =============================================================================
# 5. Event_based_Verification - HSS/ICME-based Model Evaluation
# =============================================================================
class Event_based_verification:
    """
    Performs HSS event detection and event-based performance analysis.
    - Supports stream interaction region (SIR) peak detection
    - Annotates performance over solar wind structures
    """

    def __init__(self, model_list=[]):
        self.model_list = model_list  # e.g., ['wsa_speeds', 'eswf3_2', 'model_speeds', 'persistence27']
        self.omni_peaks_list = []
        self.dl_model_peaks_list = []
        self.wsa_enlil_peaks_list = []
        self.eswf3_2_peaks_list = []
        self.persistence_peaks_list = []
        self.sr_peaks_list = []

    # we evaluate models for 6 metrics: POD, FNR, SR, FAR, CSI, and BS
    def POD(self, TP, FP, FN):
        return TP / (TP + FN) if (TP + FN) != 0 else np.nan

    def FNR(self, TP, FP, FN):
        return FN / (TP + FN) if (TP + FN) != 0 else np.nan

    def SR(self, TP, FP, FN):
        return TP / (TP + FP) if (TP + FP) != 0 else np.nan

    def FAR(self, TP, FP, FN):
        return FP / (TP + FP) if (TP + FP) != 0 else np.nan

    def CSI(self, TP, FP, FN):
        return TP / (TP + FP + FN) if (TP + FP + FN) != 0 else np.nan

    def BS(self, TP, FP, FN):
        return (TP + FP) / (TP + FN) if (TP + FN) != 0 else np.nan

    def event_verification(self, TP, FP, FN):
        return np.round(self.POD(TP, FP, FN), 2), np.round(self.FNR(TP, FP, FN), 2), \
                np.round(self.SR(TP, FP, FN), 2), np.round(self.FAR(TP, FP, FN), 2), \
                np.round(self.CSI(TP, FP, FN), 2), np.round(self.BS(TP, FP, FN), 2)
    
    def get_icme_events(self) -> list:
        return fetch_icme_events()

    def detect_HSE_blocks(self, Time: pd.Series, Speed: pd.Series):
        # (1) Mark all time points which are more than 50 km/s faster than 1 day earlier.
        candidates = [
            i for i in range(2, len(Time))
            if Speed.iloc[i] > Speed.iloc[i-2] + 50
            and not Time.iloc[i-2:i+1].isna().any()
        ]
        
        # (2) Eliminate any isolated single data points which are marked. 
        # (3) Group each contiguous block of marked points as a distinct high speed enhancement (HSE) 
        #     and find the start and end time of each HSE.
        groups = []
        if candidates:
            current = [candidates[0]]
            for idx in candidates[1:]:
                if idx == current[-1] + 1:
                    current.append(idx)
                else:
                    if len(current) > 1:
                        groups.append(current)
                    current = [idx]
            if len(current) > 1:
                groups.append(current)
        
        event_groups_dict = {'dates': [], 'speeds': [], 'index': []}
        for grp in groups:
            event_groups_dict['dates'].append([Time.iloc[i] for i in grp])
            event_groups_dict['speeds'].append([Speed.iloc[i] for i in grp])
            event_groups_dict['index'].append(grp)

        # (4) For each HSE, find the minimum speed starting 2 days ahead of the HSE till the start of the HSE, 
        #     and mark it as the minimum speed (Vmin) of the HSE; find the maximum speed starting 
        #     from the beginning of the HSE through 1 day after the HSE and mark it as the maximum speed (Vmax) of the HSE. 
        SIR_dict = {'dates': [], 'speeds': [], 'index': []}
        for i_list in event_groups_dict['index']:
            start_idx, end_idx = i_list[0], i_list[-1]
            start_idx = max(start_idx, 4)

            # Find v_min
            seg_min = Speed.iloc[start_idx-4:start_idx+1]
            argmin_index = np.argmin(seg_min)
            min_index = start_idx-4 + argmin_index
            min_date, min_speed = Time.iloc[start_idx-4 + argmin_index], seg_min.iloc[argmin_index]

            # Find v_max
            seg_max = Speed.iloc[end_idx:end_idx+3]
            argmax_index = np.argmax(seg_max)
            max_index = end_idx + argmax_index
            max_date, max_speed = Time.iloc[end_idx + argmax_index], seg_max.iloc[argmax_index]

            # (5) For each HSE, find the last time reaching Vmin and the first time reaching Vmax 
            #     and mark them as the start and end time of an SIR.
            SIR_dict['dates'].append([min_date, max_date])
            SIR_dict['speeds'].append([min_speed, max_speed])
            SIR_dict['index'].append([min_index, max_index])

        events = list(zip(SIR_dict['dates'], SIR_dict['speeds'], SIR_dict['index']))
        # (6) Combine SIRs separated by less than 0.75 day and eliminate any repeated count of SIRs.
        merged = []
        dt = timedelta(hours=18)

        for dates, speed, idx in events:
            if merged and dates[0] - merged[-1][0][1] <= dt:
                # less than 0.75 day
                merged[-1][0][1] = dates[1]
                merged[-1][1].append(speed)
                merged[-1][2].append(idx)
            else:
                merged.append([[dates[0], dates[1]], [speed], [idx]])

        merged_SIR_dict = {
            'dates': [m[0] for m in merged],
            'speeds': [m[1] for m in merged],
            'index':  [m[2] for m in merged],
        }

        merged_SIR_speed_list = []
        for sir in merged_SIR_dict['speeds']:
            if len(sir) > 1:
                merged_SIR_speed_list.append([sir[0][0], sir[1][1]])
            else: 
                merged_SIR_speed_list.append(sir[0])
        merged_SIR_dict['speeds'] = merged_SIR_speed_list

        merged_SIR_index_list = []
        for sir in merged_SIR_dict['index']:
            if len(sir) > 1:
                merged_SIR_index_list.append([sir[0][0], sir[1][1]])
            else: 
                merged_SIR_index_list.append(sir[0])
        merged_SIR_dict['index'] = merged_SIR_index_list

        merged2_SIR_dict = {'dates': [], 'speeds': [], 'index': []}
        for dates, speed, idx in zip(
            merged_SIR_dict['dates'], 
            merged_SIR_dict['speeds'], 
            merged_SIR_dict['index']
        ):
            if idx[1] - idx[0] > 2:
                merged2_SIR_dict['dates'].append(dates)
                merged2_SIR_dict['speeds'].append(speed)
                merged2_SIR_dict['index'].append(idx)

        # (9) Reject any SIRs with Vmin faster than 500 km/s, or Vmax slower than 400 km, or speed increase less than 100 km/s. 
        regrouped_SIR_dict = {'dates': [], 'speeds': [], 'index': []}
        for i in range(len(merged2_SIR_dict['index'])):
            SIR_start, SIR_end = merged2_SIR_dict['index'][i]
            SIR_speeds_list = Speed.iloc[SIR_start: SIR_end+1]
            max_index = SIR_start + np.nanargmax(SIR_speeds_list)
            v_min, v_max = np.min(SIR_speeds_list), np.max(SIR_speeds_list)
            if v_min >= 500 and v_max <= 400 and v_max-v_min < 100:
                continue
            else:
                regrouped_SIR_dict['dates'].append(merged2_SIR_dict['dates'][i])
                regrouped_SIR_dict['speeds'].append(merged2_SIR_dict['speeds'][i])
                regrouped_SIR_dict['index'].append(merged2_SIR_dict['index'][i])
        
        return regrouped_SIR_dict
    
    def plot_models_profile(self, df, SR_models, years: list, save_figure: bool, save_dir: str = "."):        
        for year in years:
            hour_range = [0, 6, 12, 18]
            year_df = df[(df['datetime'].dt.year == year) & (df['datetime'].dt.hour.isin(hour_range))]

            time = year_df['datetime']
            omni_speed = year_df['speed']

            plt.figure(figsize=(30, 5))
            plt.plot(time, omni_speed, c='black', label='OMNI', linewidth=3.5)
            # OMNI HSE events
            omni_SIR_dict = self.detect_HSE_blocks(time, omni_speed)
            for i in range(len(omni_SIR_dict['index'])):
                SIR_start, SIR_end = omni_SIR_dict['index'][i]
                SIR_speeds_list = omni_speed.iloc[SIR_start: SIR_end+1]
                max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                plt.plot(time.iloc[max_index], omni_speed.iloc[max_index], color='black', marker='v', markersize=14)
                self.omni_peaks_list.append(time.iloc[max_index])

            if 'wsa_speeds' in self.model_list:
                wsa_enlil_speed = year_df['wsa_speeds']
                plt.plot(time, wsa_enlil_speed, c='deepskyblue', label='WSA-Enlil', linestyle='--', linewidth=2, alpha=0.8)
                # WSA-ENLIL HSE events
                wsa_enlil_SIR_dict = self.detect_HSE_blocks(time, wsa_enlil_speed)
                for i in range(len(wsa_enlil_SIR_dict['index'])):
                    SIR_start, SIR_end = wsa_enlil_SIR_dict['index'][i]
                    SIR_speeds_list = wsa_enlil_speed.iloc[SIR_start: SIR_end+1]
                    max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                    plt.plot(time.iloc[max_index], wsa_enlil_speed.iloc[max_index], color='deepskyblue', marker='v', markersize=14, alpha=0.8)
                    self.wsa_enlil_peaks_list.append(time.iloc[max_index])

            if 'eswf3_2' in self.model_list:
                eswf3_2_speed = year_df['eswf3_2']
                plt.plot(time, eswf3_2_speed, c='g', label='ESWF3.2', linestyle='--', linewidth=2, alpha=0.8)
                # ESWF 3.2 HSE events
                eswf3_2_SIR_dict = self.detect_HSE_blocks(time, eswf3_2_speed)
                for i in range(len(eswf3_2_SIR_dict['index'])):
                    SIR_start, SIR_end = eswf3_2_SIR_dict['index'][i]
                    SIR_speeds_list = eswf3_2_speed.iloc[SIR_start: SIR_end+1]
                    max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                    plt.plot(time.iloc[max_index], eswf3_2_speed.iloc[max_index], color='g', marker='v', markersize=14, alpha=0.8)
                    self.eswf3_2_peaks_list.append(time.iloc[max_index])

            if 'model_speeds' in self.model_list:
                dl_model_speed = year_df['model_speeds']
                plt.plot(time, dl_model_speed, c='blue', label='DL model', linestyle='--', linewidth=2, alpha=0.8)
                # DL model HSE events
                dl_model_SIR_dict = self.detect_HSE_blocks(time, dl_model_speed)
                for i in range(len(dl_model_SIR_dict['index'])):
                    SIR_start, SIR_end = dl_model_SIR_dict['index'][i]
                    SIR_speeds_list = dl_model_speed.iloc[SIR_start: SIR_end+1]
                    max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                    plt.plot(time.iloc[max_index], dl_model_speed.iloc[max_index], color='blue', marker='v', markersize=14, alpha=0.8)
                    self.dl_model_peaks_list.append(time.iloc[max_index])

            if 'persistence27' in self.model_list:
                persist27_speed = year_df['persistence27']
                plt.plot(time, persist27_speed, c='orange', label='27-days Persistence', linestyle='--', linewidth=2, alpha=0.8)
                # 27 days Persistence model HSE events
                persist27_SIR_dict = self.detect_HSE_blocks(time, persist27_speed)
                for i in range(len(persist27_SIR_dict['index'])):
                    SIR_start, SIR_end = persist27_SIR_dict['index'][i]
                    SIR_speeds_list = persist27_speed.iloc[SIR_start: SIR_end+1]
                    max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                    plt.plot(time.iloc[max_index], persist27_speed.iloc[max_index], color='orange', marker='v', markersize=14, alpha=0.8)
                    self.persistence_peaks_list.append(time.iloc[max_index])
            
            # Symbolic Regression model
            """
            Change the SR model in these code.
            
            """
            available_colors = [
                '#FF0000',  # pure red
                '#8B00FF',  # electric purple
                '#00FFEF',  # neon cyan
                '#CCFF00',  # neon lime
                '#FF00FF',  # magenta
            ]

            for idx, SR_model in enumerate(SR_models):
                color = available_colors[idx % len(available_colors)]
                SR_speed = year_df[SR_model]
                plt.plot(time, SR_speed, c=color, label=SR_model, linewidth=3.5, alpha=0.8)
                # Symbolic Regression model HSE events
                SR_SIR_dict = self.detect_HSE_blocks(time, SR_speed)
                for i in range(len(SR_SIR_dict['index'])):
                    SIR_start, SIR_end = SR_SIR_dict['index'][i]
                    SIR_speeds_list = SR_speed.iloc[SIR_start: SIR_end+1]
                    max_index = SIR_start + np.nanargmax(SIR_speeds_list)
                    plt.plot(time.iloc[max_index], SR_speed.iloc[max_index], color=color, marker='v', markersize=14, alpha=0.8)
                    self.sr_peaks_list.append(time.iloc[max_index])

            # ICME event (gray shading)
            icme_events = []
            start_date, end_date = time.iloc[0], time.iloc[-1]
            icme_event_list = self.get_icme_events()
            for icme in icme_event_list:
                icme_start, icme_end = icme
                icme_start = datetime.strptime(icme_start, '%Y-%m-%dT%H:%M:%S')
                icme_end = datetime.strptime(icme_end, '%Y-%m-%dT%H:%M:%S')
                    
                if start_date <= icme_start <= end_date and start_date <= icme_end <= end_date:
                    icme_events.append([icme_start, icme_end])

            if len(icme_events) > 0:
                for icme_event in icme_events:
                    icme_start, icme_end = icme_event
                    plt.axvspan(icme_start, icme_end, facecolor='gray', alpha=0.4)

            extra = time.iloc[-1] + pd.Timedelta(hours=6)
            extended_time = pd.concat([time, pd.Series([extra])], ignore_index=True)
            xtick_dates = [t for t in extended_time if (t.month in [10, 11, 12, 1] and t.day == 1 and t.hour == 0)]
            xtick_labels = [time_obj.strftime("%b-%Y") for time_obj in xtick_dates]
            plt.xticks(ticks=xtick_dates, labels=xtick_labels, fontsize=24)
            plt.yticks([400, 600, 800], fontsize = 24)
            plt.xlabel('Date [month-year]', fontsize = 28, labelpad = 10)
            plt.ylabel('Speed [km/s]', fontsize = 28, labelpad = 10)

            plt.xlim(extended_time.iloc[0], extended_time.iloc[-1])
            plt.ylim(250, 900)

            # Directory to save a figure
            if save_figure:
                title_str = f"SR Model Comparisons of Slow-to-Fast Stream Interactions in {year}"
                save_path = os.path.join(save_dir, f"{title_str}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()

    def get_metrics_for_HSS_events(self):
        #dl_model_TP = 0 
        wsa_enlil_TP, eswf3_2_TP, persist27_TP, SR_TP = 0, 0, 0, 0

        for omni_peak in self.omni_peaks_list:
            omni_peak_start = omni_peak - timedelta(hours=24)
            omni_peak_end = omni_peak + timedelta(hours=24)
            
            #for model_peak in self.dl_model_peaks_list:
            #    if omni_peak_start <= model_peak <= omni_peak_end:
            #        dl_model_TP += 1
            #        break

            for wsa_peak in self.wsa_enlil_peaks_list:
                if omni_peak_start <= wsa_peak <= omni_peak_end:
                    wsa_enlil_TP += 1
                    break

            for eswf_peak in self.eswf3_2_peaks_list:
                if omni_peak_start <= eswf_peak <= omni_peak_end:
                    eswf3_2_TP += 1
                    break

            for persistence_peak in self.persistence_peaks_list:
                if omni_peak_start <= persistence_peak <= omni_peak_end:
                    persist27_TP += 1
                    break

            for sr_peak in self.sr_peaks_list:
                if omni_peak_start <= sr_peak <= omni_peak_end:
                    SR_TP += 1
                    break

        #dl_model_FP, dl_model_FN = len(self.dl_model_peaks_list) - dl_model_TP, len(self.omni_peaks_list) - dl_model_TP
        wsa_enlil_FP, wsa_enlil_FN = len(self.wsa_enlil_peaks_list) - wsa_enlil_TP,   len(self.omni_peaks_list) - wsa_enlil_TP
        eswf3_2_FP, eswf3_2_FN     = len(self.eswf3_2_peaks_list) - eswf3_2_TP,       len(self.omni_peaks_list) - eswf3_2_TP
        persist27_FP, persist27_FN = len(self.persistence_peaks_list) - persist27_TP, len(self.omni_peaks_list) - persist27_TP
        SR_FP, SR_FN               = len(self.sr_peaks_list) - SR_TP,                 len(self.omni_peaks_list) - SR_TP

        #print(dl_model_TP, dl_model_FP, dl_model_FN)
        #print(self.event_verification(dl_model_TP, dl_model_FP, dl_model_FN), '\n')

        print("WSA-ENLIL results:")
        print(wsa_enlil_TP, wsa_enlil_FP, wsa_enlil_FN)
        print(self.event_verification(wsa_enlil_TP, wsa_enlil_FP, wsa_enlil_FN), '\n')

        print("ESWF 3.2 results:")
        print(eswf3_2_TP, eswf3_2_FP, eswf3_2_FN)
        print(self.event_verification(eswf3_2_TP, eswf3_2_FP, eswf3_2_FN), '\n')

        print("27-days Persistience results:")
        print(persist27_TP, persist27_FP, persist27_FN)
        print(self.event_verification(persist27_TP, persist27_FP, persist27_FN), '\n')

        print("Symbolic Regression results:")
        print(SR_TP, SR_FP, SR_FN)
        print(self.event_verification(SR_TP, SR_FP, SR_FN))
