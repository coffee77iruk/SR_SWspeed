# 🌞 SR_SWspeed

**A Data-Driven Empirical Formula for Solar Wind Speed Prediction Using Symbolic Regression**

> 📝 **Status:** the paper is currently **under revision at *The Astrophysical Journal Supplement Series* (ApJS)**.
> A DOI is not available yet — this section will be updated with the full citation once it's assigned.

---

## 🔭 Overview

High-speed solar wind streams from coronal holes (CHs) drive recurrent geomagnetic
disturbances, and forecasting them in advance is a core space-weather problem. This
project uses **symbolic regression (PySR)** to derive a compact, interpretable
empirical formula that predicts solar wind speed at 1 AU from two EUV-derived
coronal-hole indices:

- **$A_{CH}$** — fractional coronal-hole area within a central meridional slice
- **$P_{CH}$** — a brightness-weighted coronal-hole index (sum of reciprocal pixel intensities)

both measured from SDO/AIA 193 Å / 211 Å images, combined with 27-day (1 Carrington
rotation) persistence of the observed solar wind speed.

The resulting formula is compared against three established baselines — **ESWF**,
**WSA-ENLIL**, and 27-day persistence — over 2010–2024 (OMNI 1-hour data, ICME
periods excluded), both as a continuous time series and as a high-speed-event
(HSS) detection problem.

## 📊 Key result

Entire test period (2010–2024, Oct–Dec months, ICME periods excluded):

| Model | MAE [km/s] | RMSE [km/s] | CC | DTW_mean [km/s] |
|---|---|---|---|---|
| **SR-derived formula** | **60.0** | **78.4** | **0.55** | **28.9** |
| 27-day persistence | 70.9 | 96.2 | 0.46 | 31.9 |
| ESWF | 82.3 | 108.9 | 0.38 | 37.2 |
| WSA-ENLIL | 91.3 | 120.3 | 0.36 | 54.8 |

DTW_mean (mean Dynamic Time Warping distance, Samara et al. 2022; Edward-Inatimi
et al. 2026 -- reported here as cost-per-matched-pair, i.e. the raw DTW cost
divided by the warping path's length rather than the raw point count, so it's
on the same km/s scale as MAE/RMSE without diluting singularities, over a
+-2-day Sakoe-Chiba window following Samara et al. (2022)'s own EUHFORIA
window) captures whether the *shape*/timing of predicted speed enhancements
matches observations, not just pointwise error — the SR-derived formula has
the lowest (best) DTW_mean of all four models.

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
│   ├── 05_verify_performance.ipynb    # Table 2 + Taylor diagram: SR vs. baselines
│   └── 06_hss_event_detection.ipynb   # HSS/SIR event-based verification
├── src/
│   ├── data/         # FITS calibration, CH extraction, benchmark model loaders
│   ├── models/        # PySR config, corotation projection, ensemble variants
│   ├── evaluation/     # metrics (MAE/RMSE/CC/DTW), event detection & scoring
│   ├── viz/            # all figure-generation code
│   └── utils/          # ICME masking, CH preprocessing, shared helpers
└── figures/            # published figure PNGs
```

## ⚙️ Setup

```bash
conda activate venv        # see requirements.txt for the full dependency list
```

> 🚧 `requirements.txt` is still being finalized — check back soon for the pinned
> dependency list.

## 🚀 Pipeline

`notebooks/01`–`06` walk through the full methodology interactively: AIA level-1.5
calibration, $A_{CH}$/$P_{CH}$ extraction, latitude-band comparison, SR input feature
importance, and the performance verification (Table 2, Taylor diagram) and
HSS event-detection results used in the paper.

## 📖 Citation

```
Seungwoo Ahn, Youngjae Kim, Mingyu Jeon, Hyun-Jin Jeong, Daeil Kim, Junmu Youn,
and Yong-Jae Moon, "A Data-Driven Empirical Formula for Solar Wind Speed
Prediction Using Symbolic Regression," The Astrophysical Journal Supplement
Series (in revision).
```

A full citation with DOI will be added once the paper is accepted and published.
