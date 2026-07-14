"""
Parametric (random) ensemble axis: fractional Gaussian noise on a subset of
{a, b, alpha, beta, v0}, the rest held fixed at DEFAULT_PARAMS.

Per the Step 1 sensitivity analysis (notebooks/ensemble_model/01_parameter_sensitivity.ipynb),
alpha and beta are too sensitive to randomize safely at any reasonable scale.

That leaves a, b, v0 for the two additive terms:
    v_t = a * (A_CH * P_CH)^alpha + b * (v_persist27 * v0)^beta
With alpha and beta fixed, b and v0 only ever appear as the product
b * sqrt(v0) (beta=0.5) - they are *exactly* degenerate: perturbing both
adds no extra degree of freedom over perturbing just one of them, it only
inflates the variance of the same single effective scale factor on the
persistence term. So the two independent, non-degenerate knobs are:
  - a     : scales the CH term (A_CH * P_CH)^alpha
  - v0    : scales the persistence term (v_persist27 * v0)^beta
b is left fixed at its base value to avoid double-counting that degree of freedom.
"""

import numpy as np

from .formula import DEFAULT_PARAMS, FEATURE_COLS, generalized_formula

FIXED_PARAMS = {k: v for k, v in DEFAULT_PARAMS.items() if k != "v0"}

# The two non-degenerate free parameters (see module docstring): a for the CH
# term, v0 for the persistence term. b is deliberately excluded (degenerate
# with v0 given beta fixed), alpha/beta stay fixed (Step 1: far too sensitive).
FREE_PARAMS_A_V0 = ("a", "v0")


def sample_members(df, sigma_fracs: dict, n_members: int = 300,
                    base_params: dict = DEFAULT_PARAMS, seed: int = 0) -> np.ndarray:
    """
    Sample p ~ base_params[p] * (1 + N(0, sigma_fracs[p])) independently for
    each parameter name in sigma_fracs; every other parameter in base_params
    is held fixed at its base value. Returns an (n_members, len(df)) array.
    """
    rng = np.random.default_rng(seed)
    a_ch = df[FEATURE_COLS["a_ch"]].to_numpy()
    p_ch = df[FEATURE_COLS["p_ch"]].to_numpy()
    persist = df[FEATURE_COLS["persist"]].to_numpy()

    samples = {
        p: (base_params[p] * (1.0 + rng.normal(0, sigma_fracs[p], size=n_members))
            if p in sigma_fracs else np.full(n_members, base_params[p]))
        for p in base_params
    }

    members = np.empty((n_members, len(df)))
    for i in range(n_members):
        params_i = {p: samples[p][i] for p in base_params}
        members[i] = generalized_formula(a_ch, p_ch, persist, **params_i)
    return members


def sample_v0_members(df, sigma_frac: float, n_members: int = 300,
                       v0_base: float = DEFAULT_PARAMS["v0"],
                       fixed_params: dict = FIXED_PARAMS, seed: int = 0) -> np.ndarray:
    """v0-only preview ensemble (kept for 04_crps_evaluation.ipynb). See sample_members()
    for the general (e.g. a+v0) version used in 01_parameter_sensitivity.ipynb."""
    base_params = dict(fixed_params, v0=v0_base)
    return sample_members(df, {"v0": sigma_frac}, n_members=n_members, base_params=base_params, seed=seed)
