# 🌞 SR_SWspeed

**A Data-Driven Empirical Formula for Solar Wind Speed Prediction Using Symbolic Regression**

> 📝 **Status:** the paper is currently **under revision at *The Astrophysical Journal Supplement Series* (ApJS)**.
> A DOI is not available yet — this section will be updated with the full citation once it's assigned.

---

## 🔭 Overview

High-speed solar wind streams from coronal holes (CHs) drive recurrent geomagnetic
disturbances with a periodicity of approximately 27 days. This repo applies **symbolic
regression (SR)**, via [PySR](https://github.com/MilesCranmer/PySR), to derive a
compact, interpretable formula that predicts solar wind speed at 1 AU from two
EUV-derived CH parameters, the fractional CH area ($A_{CH}$) and CH contrast
($P_{CH}$), both derived from SDO/AIA 193/211 Å images, together with prior solar
wind speeds. The model is trained on 2010-2024 OMNI data (January-September) and
evaluated on a held-out test set (October-December each year), with ICME periods
excluded.

The final formula,

$$v_t = \sqrt{A_{CH60,\,t-4d} \times P_{CH30,\,t-4d}} + \sqrt{v_{t-27d} \times 372}\ \text{km/s}$$

was selected from PySR's complexity/loss Pareto front as the simplest candidate that
combines both $A_{CH}$ and $P_{CH}$, balancing accuracy against expression simplicity
(see `notebooks/05_verify_performance.ipynb`'s Table 1 for the full candidate list).
It is compared against three baseline models, **WSA-ENLIL**, **ESWF**, and 27-day
persistence, plus an **average** (training-set-mean) baseline, both as a continuous
time series and as a high-speed-event (HSS) detection problem.

## 📊 Key result

Entire test period (2010–2024, October–December each year, ICME periods excluded):

| Model | MAE [km/s] | RMSE [km/s] | CC | SSF_mean | SSF_27days |
|---|---|---|---|---|---|
| **SR-derived formula** | **60.0** | **78.4** | **0.55** | **0.574** | **0.911** |
| WSA-ENLIL | 91.3 | 120.3 | 0.36 | 0.981 | 1.556 |
| ESWF | 82.3 | 108.9 | 0.38 | 0.722 | 1.145 |
| 27-day persistence | 70.9 | 96.2 | 0.46 | 0.630 | 1.000 |
| Average (training-set mean) | 74.8 | 93.5 | – | 1.004 | 1.592 |

The SR-derived formula outperforms all four baseline models in MAE and RMSE, and
outperforms WSA-ENLIL/ESWF/persistence in CC, across every solar cycle phase and the
entire period. The average baseline beats WSA-ENLIL and ESWF in MAE/RMSE despite
predicting a constant speed per phase, a reminder that pointwise error alone doesn't
imply real predictive skill. SSF (Sequence Similarity Factor, Samara et al. 2022)
addresses this by scoring each model's dynamic-time-warping alignment cost against
OMNI, normalized against a reference (SSF_mean: the test period's own mean speed;
SSF_27days: the persistence baseline, so persistence's own SSF_27days is
exactly 1.000 by construction), the SR-derived formula achieves the lowest SSF under
both references.

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
