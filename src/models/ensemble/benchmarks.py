"""
Shared loader for the OMNI/ESWF/WSA-ENLIL benchmark comparison used by
local-only supplementary speed-profile figures. Kept separate from
formula.py since it pulls in the repo's benchmark data modules (external
local files + a network fetch for the WSA-ENLIL Carrington-rotation table).
"""

import os

from .formula import load_full_series

_ESWF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "external", "ESWF_3_2_2012_2023.txt")
)


def load_full_series_with_benchmarks(extra_cols: list = None, eswf_path: str = _ESWF_PATH):
    """
    load_full_series() merged with ESWF3.2 ('eswf3_2' column, local file, no
    coverage after 2023-07) and WSA-ENLIL ('wsa_enlil' column, local CR files +
    a network fetch of the Carrington-rotation table, Oct-Dec only, ~10-15s).

    Requires the caller to have already added the repo's src/ directory to
    sys.path.
    """
    from data.benchmark.empirical_model.eswf3_2 import eswf32_from_file
    from data.benchmark.wsa_enlil.wsa_enlil_ccmc import WSA_ENLIL

    full_df = load_full_series(extra_cols=extra_cols)
    eswf_df = eswf32_from_file(full_df, eswf_path)[["datetime", "eswf3_2"]]
    wsa_df = WSA_ENLIL()[["datetime", "speed"]].rename(columns={"speed": "wsa_enlil"})

    return full_df.merge(eswf_df, on="datetime", how="left").merge(wsa_df, on="datetime", how="left")
