"""
Structural ensemble axes for the SR-derived formula: discrete alternative
input columns rather than randomized parameters. Each "member" here is a
deterministic prediction from a different plausible choice of which CH
columns feed the formula - not a noise sample.

Axis 2 - latitude band: which (A_CH, P_CH) area/permeability band combination
    to use (30 deg vs 60 deg), holding delay fixed at the base 4-day lag.
Axis 3 - delay: which propagation-delay lag to use for the base A_CH60/P_CH30
    combination (3d..5d), holding the latitude combination fixed.

Both axes use a=b=1, alpha=beta=0.5, v0=DEFAULT_PARAMS['v0'] (the fixed-formula
base params) - only the input columns change, per member.
"""

from .formula import DEFAULT_PARAMS, generalized_formula

# (a_ch_col, p_ch_col, label) - channels fixed at 193 (A_CH) / 211 (P_CH), delay
# fixed at the base 4-day lag; only the latitude band (30/60 deg) varies. +/-90
# excluded per the original ensemble spec (weak polar-CH / Earth connectivity).
LATITUDE_COMBOS = [
    ("A_CH30_193_lag4", "P_CH30_211_lag4", "A30-P30"),
    ("A_CH60_193_lag4", "P_CH30_211_lag4", "A60-P30 (base)"),
    ("A_CH30_193_lag4", "P_CH60_211_lag4", "A30-P60"),
    ("A_CH60_193_lag4", "P_CH60_211_lag4", "A60-P60"),
]

LATITUDE_EXTRA_COLS = sorted({col for a, p, _ in LATITUDE_COMBOS for col in (a, p)})

# Full 3x3 grid including +/-90 deg, to empirically test whether excluding it
# was the right call (rather than just asserting it from physical reasoning).
LATITUDE_BANDS = (30, 60, 90)
LATITUDE_COMBOS_FULL = [
    (
        f"A_CH{a}_193_lag4", f"P_CH{p}_211_lag4",
        f"A{a}-P{p}" + (" (base)" if (a, p) == (60, 30) else ""),
    )
    for a in LATITUDE_BANDS for p in LATITUDE_BANDS
]

LATITUDE_EXTRA_COLS_FULL = sorted({col for a, p, _ in LATITUDE_COMBOS_FULL for col in (a, p)})

# Reproduces the ad hoc CH-term uncertainty band from 05_verifing_performance.ipynb /
# 06_HSS_events_verification.ipynb (min/max of sqrt(A_CH*P_CH) over *all* A/P lag4
# columns): unlike LATITUDE_COMBOS(_FULL) above, this also lets the P_CH channel
# vary between 193 and 211, not just the latitude band. (A_CH only exists at
# channel 193 in this dataset, so only P_CH's channel can vary.) This is a wider,
# less physically-scoped sweep than the latitude-only axis - channel choice was
# never a defined ensemble axis, it's mixed in here only for comparison.
_P_CHANNELS = (193, 211)


def _channel_mixed_combos(bands):
    return [
        (
            f"A_CH{a}_193_lag4", f"P_CH{p}_{p_ch}_lag4",
            f"A{a}-P{p}@{p_ch}" + (" (base)" if (a, p, p_ch) == (60, 30, 211) else ""),
        )
        for a in bands for p in bands for p_ch in _P_CHANNELS
    ]


LATITUDE_COMBOS_CHANNEL_MIXED = _channel_mixed_combos((30, 60))
LATITUDE_COMBOS_CHANNEL_MIXED_EXTRA_COLS = sorted(
    {col for a, p, _ in LATITUDE_COMBOS_CHANNEL_MIXED for col in (a, p)}
)

LATITUDE_COMBOS_CHANNEL_MIXED_FULL = _channel_mixed_combos(LATITUDE_BANDS)
LATITUDE_COMBOS_CHANNEL_MIXED_FULL_EXTRA_COLS = sorted(
    {col for a, p, _ in LATITUDE_COMBOS_CHANNEL_MIXED_FULL for col in (a, p)}
)

# (lag_suffix, label, hours) - latitude/channel combination fixed at the base
# A_CH60_193 / P_CH30_211; only the propagation delay varies.
DELAY_STEPS = [
    ("lag3",   "3d",       72),
    ("lag3p5", "3.5d",     84),
    ("lag4",   "4d (base)", 96),
    ("lag4p5", "4.5d",    108),
    ("lag5",   "5d",       120),
]

_BASE_A_CH_CHANNEL = "A_CH60_193"
_BASE_P_CH_CHANNEL = "P_CH30_211"

DELAY_EXTRA_COLS = sorted({
    f"{channel}_{suffix}"
    for suffix, _, _ in DELAY_STEPS
    for channel in (_BASE_A_CH_CHANNEL, _BASE_P_CH_CHANNEL)
})


def build_latitude_members(df, persist_col: str = "speed_p27", params: dict = DEFAULT_PARAMS,
                            combos: list = LATITUDE_COMBOS) -> dict:
    """Return {label: v_t array} for each latitude-band combination in `combos`
    (defaults to the 4-combo 30/60-only grid; pass LATITUDE_COMBOS_FULL for the
    3x3 grid that also includes +/-90 deg)."""
    persist = df[persist_col]
    return {
        label: generalized_formula(df[a_ch_col], df[p_ch_col], persist, **params)
        for a_ch_col, p_ch_col, label in combos
    }


def build_delay_members(df, persist_col: str = "speed_p27", params: dict = DEFAULT_PARAMS) -> dict:
    """Return {label: v_t array} for each of the 5 propagation-delay choices."""
    persist = df[persist_col]
    return {
        label: generalized_formula(
            df[f"{_BASE_A_CH_CHANNEL}_{suffix}"], df[f"{_BASE_P_CH_CHANNEL}_{suffix}"], persist, **params
        )
        for suffix, label, _ in DELAY_STEPS
    }
