# 🌞 SR_SWspeed

**A Data-Driven Empirical Formula for Solar Wind Speed Prediction Using Symbolic Regression**

> 📝 **Status:** the paper is currently **under revision at *The Astrophysical Journal Supplement Series* (ApJS)**.
> A DOI is not available yet — this section will be updated with the full citation once it's assigned.

---

## 🔭 Overview

High-speed solar wind streams (HSSs) from coronal holes (CHs) are known to cause
recurrent geomagnetic disturbances with a periodicity of approximately 27 days due to
the rotation rate of the Sun. This project applies **symbolic regression (SR)**, via
the [PySR](https://github.com/MilesCranmer/PySR) framework, to solar wind speed data
at 1 AU to derive a data-driven formula as a function of two CH parameters:

- **Fractional CH area ($A_{CH}$)** — $N_{CH}/N_{total}$, the number of CH pixels over
  the total number of pixels within a central meridional region of ±7.5° heliographic
  longitude (SDO/AIA 193 Å, SPoCA-segmented CH boundaries)
- **CH contrast ($P_{CH}$)** — the reciprocal sum of pixel brightness values within a
  central meridional region of ±10° heliographic longitude (SDO/AIA 193 Å or 211 Å;
  the published formula's $P_{CH30}$ term is estimated from 211 Å images)

together with prior solar wind speeds. We train the SR model on data spanning May
2010-December 2024, divided into a training set (January-September) and a test set
(October-December) each year to account for the solar cycle effect, with ICME periods
excluded.

The SR-derived formula,

$$v_t = \sqrt{A_{CH60,\,t-4d} \times P_{CH30,\,t-4d}} + \sqrt{v_{t-27d} \times 372}\ \text{km/s}$$

achieves an RMSE of 78.4 km/s and a correlation coefficient of 0.55. The first term
indicates the acceleration component governed by CHs, and the second term indicates
the background component given by the geometric mean of the 27-day persistence speed
and the typical slow solar wind speed. We compare it against three baseline models —
**WSA-ENLIL**, **ESWF**, and the 27-day persistence model — plus an **average** model
that predicts the mean observed speed of the training set for each phase, as a
baseline for the statistical comparison, both as a continuous time series and as a
high-speed-event (HSS) detection problem.

## 📊 Key result

Entire test period (2010–2024, October–December each year, ICME periods excluded):

| Model | MAE [km/s] | RMSE [km/s] | CC | SSF_mean | SSF_27days |
|---|---|---|---|---|---|
| **SR-derived formula** | **60.0** | **78.4** | **0.55** | **0.574** | **0.911** |
| 27-day persistence | 70.9 | 96.2 | 0.46 | 0.630 | 1.000 |
| ESWF | 82.3 | 108.9 | 0.38 | 0.722 | 1.145 |
| WSA-ENLIL | 91.3 | 120.3 | 0.36 | 0.981 | 1.556 |
| Average (training-set mean) | 74.8 | 93.5 | – | 1.004 | 1.592 |

The SR-derived formula shows better performance than all four baseline models across
all solar cycle phases in MAE and RMSE, achieving an MAE of 60.0 km/s and an RMSE of
78.4 km/s for the entire period. It also shows better performance than WSA-ENLIL,
ESWF, and the persistence model in CC, achieving a CC of 0.55 for the entire period.
Notably, the average model also shows lower MAE and RMSE than WSA-ENLIL and ESWF in
most phases and for the entire period, even though it predicts a constant speed for
each phase — this indicates that MAE and RMSE, as point-by-point metrics, can remain
low even for a model that captures no temporal variation.

The sequence similarity factor (SSF) evaluates how well each model captures the
temporal variation of the OMNI speed, based on dynamic time warping (DTW; Samara et
al. 2022). DTW finds the optimal alignment between the observed and predicted values
that minimizes the total cumulative cost over all possible alignment paths, applied to
the hourly time series without smoothing and using a Sakoe-Chiba band that restricts
matched points to within ±2 days of each other. Each test period is divided into
contiguous time blocks at every ICME interval and at every missing-data gap longer
than 1 hr, while gaps of 1 hr are bridged by linear interpolation and do not break a
block. SSF is then defined as

$$\mathrm{SSF} = \frac{\mathrm{DTW_{score}}(O, M)}{\mathrm{DTW_{score}}(O, R)} \in [0, \infty)$$

where $O$, $M$, and $R$ denote the observed, predicted, and reference time series,
respectively: zero for a perfect forecast, and unity when the forecast performs as
well as the reference. We use two references — the mean observed speed of the test
set for each phase (SSF_mean) and the 27-day persistence model (SSF_27days) — so
persistence's own SSF_27days is exactly 1.000 by construction, and any value below 1
means that model beats plain persistence. The SR-derived formula achieves the lowest
SSF under both references, of all four baseline models, for the entire period.

<p align="center">
  <img src="figures/figure1_ch_parameters.png" alt="A_CH and P_CH computation from SDO/AIA EUV images" width="850">
</p>

<p align="center">
  <img src="figures/figure2_speed_profiles_cr_euv.png" alt="SR-derived formula vs OMNI/ESWF/WSA-ENLIL/persistence speed profiles for representative Carrington rotations, with AIA EUV context" width="850">
</p>

<p align="center">
  <img src="figures/figure3_sr_vs_sunspot.png" alt="SR-derived formula performance vs. sunspot number across the solar cycle" width="850">
</p>

<p align="center">
  <img src="figures/figure4_performance_by_speed_range.png" alt="MAE, RMSE, and bias by OMNI solar wind speed range" width="850">
</p>

## 📁 Repository structure

```
SR_SWspeed/
├── notebooks/
│   ├── 01_convert_to_level1.5.ipynb   # AIA level 1 -> 1.5 calibration walkthrough
│   ├── 02_get_CH_parameter.ipynb      # A_CH / P_CH computation walkthrough
│   ├── 03_compare_CH_parameters.ipynb # latitude-band parameter comparison
│   ├── 04_feature_importance.ipynb    # SR input feature importance
│   ├── 05_verify_performance.ipynb    # Table 1 (formula selection) + Table 2 + Taylor diagram
│   └── 06_hss_event_detection.ipynb   # HSS/SIR event-based verification
├── src/
│   ├── data/         # FITS calibration, CH extraction, benchmark model loaders
│   ├── models/        # PySR config, corotation projection, ensemble variants
│   ├── evaluation/     # metrics (MAE/RMSE/CC/DTW/SSF), event detection & scoring
│   ├── viz/            # all figure-generation code
│   └── utils/          # ICME masking, CH preprocessing, shared helpers
└── figures/            # published figure PNGs
```

## ⚙️ Setup

```bash
conda create -n venv python=3.12
conda activate venv
pip install -r requirements.txt
```

## 🚀 Pipeline

`notebooks/01`–`06` walk through the full methodology interactively: AIA level-1.5
calibration, $A_{CH}$/$P_{CH}$ extraction, latitude-band comparison, SR input feature
importance, and the performance verification (PySR formula-selection Table 1,
Table 2, Taylor diagram) and HSS event-detection results used in the paper.

## 📖 Citation

```
Seungwoo Ahn, Youngjae Kim, Mingyu Jeon, Hyun-Jin Jeong, Daeil Kim, Junmu Youn,
and Yong-Jae Moon, "A Data-Driven Empirical Formula for Solar Wind Speed
Prediction Using Symbolic Regression," The Astrophysical Journal Supplement
Series (in revision).
```

A full citation with DOI will be added once the paper is accepted and published.
