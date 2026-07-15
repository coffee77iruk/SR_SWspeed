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

| Model | MAE [km/s] | RMSE [km/s] | CC |
|---|---|---|---|
| **SR-derived formula** | **60.0** | **78.4** | **0.55** |
| 27-day persistence | 70.9 | 96.2 | 0.46 |
| ESWF | 82.3 | 108.9 | 0.38 |
| WSA-ENLIL | 91.3 | 120.3 | 0.36 |

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
├── scripts/                     # end-to-end pipeline, run in order
│   ├── 01_get_parameters.py     #   FITS -> A_CH / P_CH CSVs
│   ├── 02_prepare_sr_data.py    #   CH params + OMNI -> phase-segmented SR datasets
│   └── 03_run_sr_model.py       #   PySR training (LOGO-CV, per solar-cycle phase)
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

## 🚀 Reproducing the pipeline

```bash
# 1. Extract A_CH / P_CH from SDO/AIA FITS files
python scripts/01_get_parameters.py --channel "193,211" --start "2010-01-01" --end "2025-01-01" \
    --base_dir "D:/Data/AIA_level1" --save_dir "../data/interim" --cores 12

# 2. Build phase-segmented SR training datasets (rising/maximum/declining/minimum/entire)
python scripts/02_prepare_sr_data.py

# 3. Train the symbolic regression model (LOGO-CV, matches the paper)
python scripts/03_run_sr_model.py --phase entire
```

Each script's own docstring has the full flag reference. `notebooks/01`–`06` walk
through the same pipeline interactively, ending with the performance verification
and HSS event-detection tables/figures used in the paper.

## 📖 Citation

```
Seungwoo Ahn, Youngjae Kim, Mingyu Jeon, Hyun-Jin Jeong, Daeil Kim, Junmu Youn,
and Yong-Jae Moon, "A Data-Driven Empirical Formula for Solar Wind Speed
Prediction Using Symbolic Regression," The Astrophysical Journal Supplement
Series (in revision).
```

A full citation with DOI will be added once the paper is accepted and published.
