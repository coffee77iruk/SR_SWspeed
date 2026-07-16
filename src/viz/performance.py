"""
Visualization helpers for SR-vs-baseline performance comparison.

plot_sr_vs_sunspot          -> paper Figure: yearly MAE/RMSE/CC vs sunspot number,
                                with solar-cycle-phase shading (published + transparent
                                ppt variant via transparent=True).
plot_binned_performance      -> paper Figure: MAE/RMSE/Bias binned by OMNI speed range,
                                with bootstrap CI (published + ppt variant).
plot_binned_performance_heatmap -> supplementary (not published): same binned stats,
                                shown as a heatmap + line plot; notebook-only, not saved
                                as a standalone figures/ deliverable.
plot_latitude_combo_heatmap  -> supplementary: RMSE/MAE/CC across all A_CH/P_CH
                                latitude-band combinations, published A60xP30 highlighted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PHASE_STYLES = {
    "Rising":    {"years": list(range(2010, 2013)) + list(range(2021, 2024)), "color": "lightskyblue"},
    "Maximum":   {"years": [2013, 2014, 2024], "color": "salmon"},
    "Declining": {"years": list(range(2015, 2019)), "color": "khaki"},
    "Minimum":   {"years": [2019, 2020], "color": "lightgray"},
}

MODEL_STYLES = {
    "speed_p27": {"label": "Persistence (27 days)", "color": "orange"},
    "wsa_enlil": {"label": "WSA-ENLIL", "color": "deepskyblue"},
    "eswf2":     {"label": "ESWF", "color": "green"},
    "best_sr":   {"label": "SR-derived formula", "color": "red"},
    "av":        {"label": "Average prediction (Collin+2025)", "color": "gray"},
}


def _interp_yearly_series(metrics_yearly_df, date, model, metric):
    """Interpolate a yearly metric value to a monthly date axis (Oct anchor)."""
    sub = metrics_yearly_df[metrics_yearly_df["model"] == model]
    year_to_val = dict(zip(sub["year"].astype(int), sub[metric].astype(float)))
    vals = [year_to_val.get(d.year) if d.month == 10 else np.nan for d in date]
    return (
        pd.Series(vals, index=pd.to_datetime(date))
        .interpolate(method="linear", limit_direction="both")
        .tolist()
    )


def plot_sr_vs_sunspot(metrics_yearly_df, date, sunspot_m, sunspot_ms,
                       choose_models=("best_sr", "eswf2", "wsa_enlil", "speed_p27"),
                       transparent=False, save_path=None):
    """3-row figure: MAE/RMSE/CC per year (twin axis) over sunspot number, with
    solar-cycle-phase background shading."""
    metrics = ["MAE", "RMSE", "CC"]
    interped = {m: {met: _interp_yearly_series(metrics_yearly_df, date, m, met) for met in metrics}
                for m in choose_models}

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharey=False)
    if transparent:
        fig.patch.set_facecolor("none")
    fontsize = 21
    years = pd.date_range("2010-01-01", "2025-01-01", freq="YS")
    model_handles, model_labels = None, None

    for row_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        if transparent:
            ax.set_facecolor("white")
        ax2 = ax.twinx()

        for style in PHASE_STYLES.values():
            for y in style["years"]:
                ax.axvspan(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01"),
                          color=style["color"], alpha=0.15, zorder=0)

        ax.plot(date, sunspot_m, c="gray", alpha=0.8, label="Monthly mean")
        ax.plot(date, sunspot_ms, c="black", alpha=0.8, label="13-month smoothed")
        if metric == "CC":
            ax.set_xlabel("Year", fontsize=fontsize)
        if row_idx == 1:
            ax.set_ylabel("Sunspot Number", fontsize=fontsize)
        ax.set_ylim(0, 250)
        ax.legend(loc="upper left", fontsize=fontsize - 4, labelspacing=0.3, handlelength=1.2)

        best_set = None
        for m in choose_models:
            yvals = interped[m][metric]
            style = MODEL_STYLES[m]
            linestyle = "-" if m == "best_sr" else "--"
            linewidth = 3 if m == "best_sr" else 2
            line, = ax2.plot(date, yvals, c=style["color"], label=style["label"],
                            linestyle=linestyle, linewidth=linewidth)
            if m == "best_sr":
                best_set = yvals

        ax2.set_ylabel(metric, fontsize=fontsize + 1)
        if row_idx == 0:
            model_handles, model_labels = ax2.get_legend_handles_labels()
        if metric in ("MAE", "RMSE"):
            ax2.set_ylim(20, 180)
            ax2.set_yticks([30, 60, 90, 120, 150, 180])
        if metric == "CC":
            ax2.set_ylim(-0.15, 1.0)
            ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])

        if best_set is not None:
            if metric in ("MAE", "RMSE"):
                target_idx = int(np.nanargmin(best_set))
                target_label = f"Min: {best_set[target_idx]:.1f} $km/s$"
            else:
                target_idx = int(np.nanargmax(best_set))
                target_label = f"Max: {best_set[target_idx]:.2f}"
            target_x, target_val = date[target_idx], best_set[target_idx]
            ax2.scatter(target_x, target_val, color="black", s=60, zorder=5, edgecolor="white")
            ax2.text(target_x, target_val + (5 if metric != "CC" else 0.05),
                    f"{target_label} ({target_x.year})", color="black", fontsize=fontsize - 4,
                    ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.25"))

        ax.set_xticks(years)
        ax.set_xticklabels([str(y.year) for y in years], rotation=0, fontsize=fontsize - 3)
        ax.tick_params(axis="y", labelsize=fontsize - 4.5)
        ax2.tick_params(axis="y", labelsize=fontsize - 4.5)
        ax.margins(x=0.01)
        ax2.margins(x=0.01)

    fig.legend(model_handles, model_labels, loc="upper center", ncol=len(model_labels),
              fontsize=fontsize - 2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(w_pad=3, rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=500, facecolor="none" if transparent else "white", bbox_inches="tight")
    return fig


def plot_binned_performance(stats_dict, bin_labels, color_map,
                            metrics=("MAE", "RMSE", "Bias"), figsize=(20, 24),
                            transparent=False, save_path=None):
    """Line plot (with bootstrap CI error bars) of binned MAE/RMSE/Bias/CC,
    one row per metric (stacked, not side-by-side) so the shared x-axis
    (speed-range bins) only needs to be drawn once, under the bottom row --
    and every row gets the full figure width instead of splitting it three
    ways."""
    models = list(stats_dict.keys())
    display_names = [m.replace("SR-derived formula", "SR-derived\nformula") for m in models]

    ref_model = models[0]
    n_vals = stats_dict[ref_model]["N"].values.astype(int)
    bin_labels_n = [f"{bl}\n({n})" for bl, n in zip(bin_labels, n_vals)]

    fig = plt.figure(figsize=figsize)
    if transparent:
        fig.patch.set_facecolor("none")
    gs = gridspec.GridSpec(len(metrics), 1, hspace=0.10, top=0.94)
    axes_line = []

    for row_idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[row_idx, 0])
        axes_line.append(ax)
        x = np.arange(len(bin_labels))

        for m, dn in zip(models, display_names):
            y = stats_dict[m][metric].values.astype(float)
            color = color_map.get(m, "gray")
            legend_label = dn.replace("\n", " ")
            lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
            yerr = None
            if lo_col in stats_dict[m].columns and hi_col in stats_dict[m].columns:
                yerr = np.array([stats_dict[m][lo_col].values.astype(float),
                                 stats_dict[m][hi_col].values.astype(float)])
            lw = 5.5 if m == "SR-derived formula" else 3.5
            ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, lw=lw, markersize=11,
                       capsize=5, capthick=1.5, elinewidth=2.2, label=legend_label)

        if metric == "Bias":
            ax.axhline(0, color="gray", lw=1.2, ls="--")
            ax.set_ylim(-320, 120)
            ax.set_yticks(np.arange(-300, 101, 100))
        elif metric in ("MAE", "RMSE"):
            ax.set_ylim(0, 330)
            ax.set_yticks(np.arange(0, 331, 50))

        ax.set_xticks(x)
        if row_idx == len(metrics) - 1:
            ax.set_xticklabels(bin_labels_n, fontsize=29)
            ax.set_xlabel("OMNI speed range [km/s]", fontsize=34, labelpad=12)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(metric, fontsize=34)
        ax.tick_params(axis="y", labelsize=29)
        ax.grid(axis="y", alpha=0.35)

    # Draw order (from `models`) controls z-order -- SR-derived formula is
    # drawn last so it layers on top of the other lines where they overlap --
    # but the legend always lists it first regardless of draw order.
    handles, labels = axes_line[0].get_legend_handles_labels()
    if "SR-derived formula" in labels:
        idx = labels.index("SR-derived formula")
        handles = [handles[idx]] + handles[:idx] + handles[idx + 1:]
        labels = [labels[idx]] + labels[:idx] + labels[idx + 1:]
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=29,
              framealpha=0, bbox_to_anchor=(0.5, 0.99))

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches="tight",
                   facecolor="none" if transparent else "white")
    return fig


def plot_binned_performance_heatmap(stats_dict, bin_labels, color_map,
                                    metrics=("MAE", "RMSE", "Bias", "CC"),
                                    figsize=(38, 15)):
    """Heatmap + line plot of binned MAE/RMSE/Bias/CC. Supplementary/exploratory
    only (not a published figure) -- not saved by default, just returned for
    inline notebook display."""
    models = list(stats_dict.keys())
    display_names = [
        m.replace("Persistence (27 days)", "27-day\nPersistence").replace("SR-derived formula", "SR-derived\nformula")
        for m in models
    ]

    ref_model = models[0]
    n_vals = stats_dict[ref_model]["N"].values.astype(int)
    bin_labels_n = [f"{bl}\n({n})" for bl, n in zip(bin_labels, n_vals)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, len(metrics), hspace=0.2, wspace=0.4)
    axes_line = []

    shared_vmin, shared_vmax = np.inf, -np.inf
    for metric in ("RMSE", "MAE"):
        if metric in metrics:
            mat = np.array([stats_dict[m][metric].values.astype(float) for m in models])
            shared_vmin = min(shared_vmin, np.nanmin(mat))
            shared_vmax = max(shared_vmax, np.nanmax(mat))

    for col_idx, metric in enumerate(metrics):
        ax_h = fig.add_subplot(gs[0, col_idx])
        data_matrix = np.array([stats_dict[m][metric].values.astype(float) for m in models])

        if metric in ("RMSE", "MAE"):
            cmap, vmin, vmax = "RdYlGn_r", shared_vmin, shared_vmax
        elif metric == "CC":
            cmap, vmin, vmax = "RdYlGn", -0.3, 0.3
        else:
            cmap, vmin, vmax = "RdBu_r", -300, 300

        im = ax_h.imshow(data_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=16)
        if metric == "Bias":
            cb.set_ticks([-300, -150, 0, 150, 300])
        if metric == "CC":
            cb.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])

        ax_h.set_xticks(range(len(bin_labels)))
        ax_h.set_xticklabels(bin_labels_n, fontsize=16)
        ax_h.set_yticks(range(len(models)))
        ax_h.set_yticklabels(display_names, fontsize=18)
        ax_h.set_title(metric, fontsize=24, pad=10)

        for r in range(len(models)):
            for c in range(len(bin_labels)):
                val = data_matrix[r, c]
                if not np.isnan(val):
                    txt = f"{val:.2f}" if metric == "CC" else f"{val:.0f}"
                    ax_h.text(c, r, txt, ha="center", va="center", fontsize=18, color="black")

        ax_l = fig.add_subplot(gs[1, col_idx])
        axes_line.append(ax_l)
        x = np.arange(len(bin_labels))

        for m, dn in zip(models, display_names):
            y = stats_dict[m][metric].values.astype(float)
            color = color_map.get(m, "gray")
            ax_l.plot(x, y, marker="o", color=color, lw=2.5, markersize=9, label=dn.replace("\n", " "))

        if metric == "Bias":
            ax_l.axhline(0, color="gray", lw=1.2, ls="--")
        elif metric == "CC":
            ax_l.set_ylim(-0.3, 1)

        ax_l.set_xticks(x)
        ax_l.set_xticklabels(bin_labels_n, fontsize=18)
        ax_l.set_xlabel("OMNI speed range [km/s]", fontsize=22, labelpad=12)
        ax_l.set_ylabel(metric, fontsize=22)
        ax_l.tick_params(axis="y", labelsize=18)
        ax_l.grid(axis="y", alpha=0.35)

    handles, labels = axes_line[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=24,
              framealpha=0, bbox_to_anchor=(0.5, 0.97))
    return fig


def plot_latitude_combo_heatmap(combo_df, phase="Entire", metrics=("RMSE", "MAE", "CC"),
                                highlight=(60, 30), a_bands=(30, 60, 90), p_bands=(30, 60, 90)):
    """
    Heatmap of RMSE/MAE/CC across every A_CH/P_CH latitude-band combination for
    one phase (output of evaluate_latitude_combinations). The published formula's
    band combo (default A60xP30) is bolded. Supplementary/exploratory -- not a
    published figure, just returned for inline notebook display.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]

    sub = combo_df[combo_df["phase"] == phase]
    a_desc = sorted(a_bands, reverse=True)

    for ax, metric in zip(axes, metrics):
        pivot = sub.pivot(index="A_band", columns="P_band", values=metric)
        pivot = pivot.reindex(index=a_bands, columns=p_bands).iloc[::-1]

        cmap = "viridis_r" if metric in ("RMSE", "MAE") else "viridis"
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(p_bands))); ax.set_xticklabels(p_bands)
        ax.set_yticks(range(len(a_bands))); ax.set_yticklabels(a_desc)
        ax.set_xlabel(r"$P_{CH}$ latitude band [deg]")
        ax.set_ylabel(r"$A_{CH}$ latitude band [deg]")
        ax.set_title(f"{metric} ({phase} period)")

        for i, a in enumerate(a_desc):
            for j, p in enumerate(p_bands):
                val = pivot.loc[a, p]
                is_best = (a, p) == highlight
                ax.text(j, i, f"{val:.1f}" if metric != "CC" else f"{val:.2f}",
                        ha="center", va="center", color="white", fontsize=12,
                        fontweight="bold" if is_best else "normal")

        fig.colorbar(im, ax=ax, label=metric)

    plt.tight_layout()
    return fig
