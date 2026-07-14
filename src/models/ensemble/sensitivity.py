"""
Per-parameter sensitivity analysis for the generalized SR formula.

For each free parameter (a, b, alpha, beta, v0), perturb it individually by
+-1%, +-5%, +-10% around its base value (others held fixed) and measure the
resulting change in predicted v_t (km/s) across the Oct-Dec test set. Mirrors
the WSA sensitivity-analysis logic in Reiss et al. (2020): this determines
which parameters need the most conservative ensemble noise scale.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .formula import DEFAULT_PARAMS, FEATURE_COLS, generalized_formula

PERTURBATION_LEVELS = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]

# dataviz skill: categorical slots 1 (blue) and 2 (aqua), fixed order, not cycled.
_SUBSET_COLORS = {"all": "#2a78d6", "ch_present": "#1baf7a"}
_SUBSET_LABELS = {"all": "All test rows", "ch_present": "CH-present rows"}


def _eval_formula(df: pd.DataFrame, params: dict) -> np.ndarray:
    return generalized_formula(
        df[FEATURE_COLS["a_ch"]], df[FEATURE_COLS["p_ch"]], df[FEATURE_COLS["persist"]],
        **params,
    )


def perturb_and_compute(df: pd.DataFrame, param_name: str, levels=PERTURBATION_LEVELS,
                         base_params=DEFAULT_PARAMS) -> dict:
    """Return {level: delta_v array} for a single parameter, others held at base."""
    v_base = _eval_formula(df, base_params)

    deltas = {}
    for level in levels:
        perturbed = dict(base_params)
        perturbed[param_name] = base_params[param_name] * (1.0 + level)
        v_perturbed = _eval_formula(df, perturbed)
        deltas[level] = v_perturbed - v_base
    return deltas


def compute_deltas(df: pd.DataFrame, params=("a", "b", "alpha", "beta", "v0"),
                    levels=PERTURBATION_LEVELS, base_params=DEFAULT_PARAMS) -> dict:
    """Return deltas[param][subset][level] = delta_v array, subset in {'all', 'ch_present'}."""
    subsets = {"all": df, "ch_present": df[df["is_ch_present"]]}

    deltas = {}
    for param in params:
        deltas[param] = {
            subset_name: perturb_and_compute(subset_df, param, levels, base_params)
            for subset_name, subset_df in subsets.items()
        }
    return deltas


def run_sensitivity_table(deltas: dict) -> pd.DataFrame:
    """Flatten compute_deltas() output into a summary DataFrame."""
    rows = []
    for param, by_subset in deltas.items():
        for subset_name, by_level in by_subset.items():
            for level, dv in by_level.items():
                rows.append({
                    "parameter": param,
                    "perturbation_pct": level * 100,
                    "subset": subset_name,
                    "n": len(dv),
                    "mean_dv": np.mean(dv),
                    "std_dv": np.std(dv),
                    "mean_abs_dv": np.mean(np.abs(dv)),
                    "median_abs_dv": np.median(np.abs(dv)),
                    "max_abs_dv": np.max(np.abs(dv)),
                })
    table = pd.DataFrame(rows).sort_values(["parameter", "perturbation_pct", "subset"]).reset_index(drop=True)
    return table


def plot_sensitivity(deltas: dict, save_path: str, params=("a", "b", "alpha", "beta", "v0"),
                      levels=PERTURBATION_LEVELS, fontsize: int = 13) -> None:
    """5-panel boxplot: one panel per parameter, Delta-v (km/s) vs perturbation %,
    split by subset (all test rows / CH-present rows)."""
    n_panels = len(params)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 5), sharey=True)

    box_width = 0.32
    tick_positions = np.arange(len(levels))

    for ax, param in zip(axes, params):
        for subset_idx, subset_name in enumerate(("all", "ch_present")):
            offset = (subset_idx - 0.5) * box_width
            data = [deltas[param][subset_name][level] for level in levels]
            bp = ax.boxplot(
                data,
                positions=tick_positions + offset,
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=_SUBSET_COLORS[subset_name], edgecolor="#52514e", linewidth=1.0, alpha=0.75),
                medianprops=dict(color="#0b0b0b", linewidth=1.5),
                whiskerprops=dict(color="#52514e", linewidth=1.0),
                capprops=dict(color="#52514e", linewidth=1.0),
            )

        ax.axhline(0, color="#c3c2b7", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{int(l * 100):+d}%" for l in levels], fontsize=fontsize - 2)
        ax.set_title(param, fontsize=fontsize + 1)
        ax.set_xlabel("Perturbation", fontsize=fontsize - 1)
        ax.tick_params(axis="y", labelsize=fontsize - 2)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=-1)

    axes[0].set_ylabel(r"$\Delta v$ [km/s]", fontsize=fontsize)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_SUBSET_COLORS[name], edgecolor="#52514e", alpha=0.75, label=_SUBSET_LABELS[name])
        for name in ("all", "ch_present")
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=fontsize - 1,
               frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.15, wspace=0.08)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
