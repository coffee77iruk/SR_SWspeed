"""
Combined ensemble: the parametric axis (a, v0 fractional Gaussian noise) and
the two structural axes (latitude band, propagation delay) sampled together,
one independent random draw of each per member. This is the standard way to
fold a parametric axis and categorical structural axes into a single joint
ensemble (cf. multi-parameter + multi-physics-scheme ensembles in NWP): each
member gets its own (eps_a, eps_v0) noise draw *and* its own random pick of
which (latitude, delay) column combination to evaluate the formula with.

alpha, beta, b stay fixed throughout (Step 1: too sensitive / degenerate with
v0 - see 01_parameter_sensitivity.ipynb). +/-90 deg is excluded here (matches
the original ensemble spec's default axis, not the channel-mixed/90-included
variants explored in 02_latitude_ensemble.ipynb).
"""

import numpy as np

from .formula import DEFAULT_PARAMS, generalized_formula
from .structural import DELAY_STEPS

_A_CHANNEL = "193"
_P_CHANNEL = "211"
LAT_DELAY_BANDS = (30, 60)

LAT_DELAY_COMBOS = [
    (f"A_CH{a}_{_A_CHANNEL}_{suffix}", f"P_CH{p}_{_P_CHANNEL}_{suffix}", f"A{a}-P{p}-{lag_label}")
    for a in LAT_DELAY_BANDS
    for p in LAT_DELAY_BANDS
    for suffix, lag_label, _ in DELAY_STEPS
]

LAT_DELAY_EXTRA_COLS = sorted({col for a, p, _ in LAT_DELAY_COMBOS for col in (a, p)})


def sample_combined_members(df, sigma_fracs: dict, n_members: int = 300,
                             combos: list = LAT_DELAY_COMBOS, base_params: dict = DEFAULT_PARAMS,
                             persist_col: str = "speed_p27", seed: int = 0):
    """
    For each of n_members: sample (eps_a, eps_v0, ...) per sigma_fracs AND an
    independent uniform-random pick of one (a_ch_col, p_ch_col) combo from
    `combos`, then evaluate generalized_formula(). Returns
    (members (n_members, len(df)) array, combo_idx (n_members,) array of
    which combo each member used).
    """
    rng = np.random.default_rng(seed)
    persist = df[persist_col].to_numpy()

    combo_arrays = [(df[a_col].to_numpy(), df[p_col].to_numpy()) for a_col, p_col, _ in combos]
    combo_idx = rng.integers(0, len(combo_arrays), size=n_members)

    samples = {
        p: (base_params[p] * (1.0 + rng.normal(0, sigma_fracs[p], size=n_members))
            if p in sigma_fracs else np.full(n_members, base_params[p]))
        for p in base_params
    }

    members = np.empty((n_members, len(df)))
    for i in range(n_members):
        a_ch, p_ch = combo_arrays[combo_idx[i]]
        params_i = {p: samples[p][i] for p in base_params}
        members[i] = generalized_formula(a_ch, p_ch, persist, **params_i)
    return members, combo_idx
