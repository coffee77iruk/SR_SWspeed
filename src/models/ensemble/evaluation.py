"""
Ensemble evaluation metrics: CRPS, spread-skill ratio, rank (Talagrand)
histogram. No external scoring dependency (properscoring is not installed) -
CRPS uses the standard order-statistics decomposition, which is exact and
O(n_members log n_members) per time step rather than the naive O(n_members^2)
pairwise sum.

CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|,  X, X' iid ~ F (the ensemble)
"""

import numpy as np


def crps_ensemble(observations, members) -> np.ndarray:
    """
    Empirical CRPS per time step for an ensemble forecast.

    observations : (T,) array of true values
    members      : (n_members, T) array

    Returns a (T,) array; NaN at any time step where the observation or any
    member is NaN (e.g. ICME-masked periods), matching the upstream NaN-masking
    convention used throughout this pipeline.
    """
    obs = np.asarray(observations, dtype=float)
    mem = np.asarray(members, dtype=float)
    n = mem.shape[0]

    term1 = np.mean(np.abs(mem - obs[None, :]), axis=0)

    sorted_mem = np.sort(mem, axis=0)
    k = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
    weights = 2 * k - n - 1
    term2 = np.sum(weights * sorted_mem, axis=0) / (n ** 2)

    return term1 - term2


def spread_skill_ratio(members, observations) -> float:
    """
    RMS ensemble spread (std across members) divided by the RMSE of the
    ensemble mean. ~1.0 indicates a well-calibrated ensemble; <1 means
    overconfident (spread too narrow), >1 means underconfident (too wide).
    """
    obs = np.asarray(observations, dtype=float)
    mem = np.asarray(members, dtype=float)

    ens_mean = np.mean(mem, axis=0)
    ens_std = np.std(mem, axis=0, ddof=1)

    valid = ~np.isnan(obs) & ~np.isnan(ens_mean)
    rmse = np.sqrt(np.mean((ens_mean[valid] - obs[valid]) ** 2))
    spread = np.sqrt(np.mean(ens_std[valid] ** 2))
    return spread / rmse


def rank_histogram(members, observations) -> np.ndarray:
    """
    Talagrand rank histogram: for each time step, count how many members are
    below the observation (0..n_members), then bin-count across all time
    steps. A flat histogram indicates a well-calibrated ensemble; U-shaped
    means too narrow (obs often falls outside the ensemble), dome-shaped means
    too wide. Returns a (n_members + 1,) count array.
    """
    obs = np.asarray(observations, dtype=float)
    mem = np.asarray(members, dtype=float)
    n_members = mem.shape[0]

    valid = ~np.isnan(obs) & ~np.isnan(mem).any(axis=0)
    ranks = np.sum(mem[:, valid] < obs[None, valid], axis=0)
    return np.bincount(ranks, minlength=n_members + 1)
