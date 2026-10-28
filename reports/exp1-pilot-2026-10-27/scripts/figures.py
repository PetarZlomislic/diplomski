"""Figures for the Experiment 1 Stage 1 pilot review (run tables.py first).

  uv run python reports/exp1-pilot-2026-10-27/scripts/figures.py

Reads reports/exp1-pilot-2026-10-27/tables/*.csv, writes figures/*.png (150 dpi, light).
Palette and chrome are the repo's (src/analysis.py); a model keeps its color everywhere.
"""
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from src.analysis import (CONFIGS, DEGRADATIONS, GRID, INK, INK_MUTED, SURFACE,  # noqa: E402
                          model_colors)

REPORT = Path(__file__).resolve().parents[1]
T, F = REPORT / "tables", REPORT / "figures"
F.mkdir(exist_ok=True)
C = model_colors(list(CONFIGS))
ARMS = list(CONFIGS)
# Sequential blue (palette.md, 100 -> 700) and blue <-> red diverging with gray midpoint.
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95",
                                                "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                     "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def title(fig, text, sub=None):
    fig.suptitle(text, x=0.01, ha="left", color=INK, fontsize=11)
    if sub:
        fig.text(0.01, 0.915, sub, ha="left", color=INK_MUTED, fontsize=8)


def save(fig, name, rect=(0, 0, 1, 0.88)):
    fig.tight_layout(rect=rect)
    fig.savefig(F / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 1 clean performance
clean = pd.read_csv(T / "clean.csv").set_index("model").loc[ARMS]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, col, lab in zip(axes, ["f1_macro", "f1_micro"], ["macro-F1", "micro-F1"]):
    y = np.arange(len(ARMS))[::-1]
    ax.barh(y, clean[col], height=0.55, color=[C[m] for m in ARMS], zorder=2)
    for yi, v in zip(y, clean[col]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=INK, fontsize=8)
    if col == "f1_macro":
        for m in ("opt", "sar"):
            r, yi = clean.loc[m, "bifold_ref"], y[ARMS.index(m)]
            ax.plot([r, r], [yi - 0.38, yi + 0.38], color=INK, linewidth=2, zorder=3)
            ax.text(r + 0.01, yi + 0.36, f"BIFOLD ResNet-18 {r:.3f}", color=INK_MUTED,
                    fontsize=7, va="bottom")
    ax.set_yticks(y, ARMS)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel(f"clean test {lab}")
    style(ax, grid="x")
title(fig, "Clean: the four fusion arms sit within 0.017 macro-F1 of opt; opt and sar are "
           "~0.10 below the BIFOLD references",
      "seed 0, test split, threshold 0.5. Black ticks: published BIFOLD ResNet-18 "
      "S2-only / S1-only test macro-F1")
save(fig, "fig_clean.png")

# ---------------------------------------------------------------- 2 curves (repo figures)
for src, dst in [("fig1_absolute.png", "fig_curves_absolute.png"),
                 ("fig2_retention.png", "fig_curves_retention.png"),
                 ("fig3_h4_interaction.png", "fig_h4_interaction.png")]:
    shutil.copyfile(REPORT / "analysis" / "exp1" / src, F / dst)

# ---------------------------------------------------------------- 3 per-class heatmaps
pc = pd.read_csv(T / "per_class_clean.csv").set_index("class")[ARMS]
labels = [textwrap.shorten(c, 42, placeholder="...") for c in pc.index]
fig, ax = plt.subplots(figsize=(7.5, 7.2))
im = ax.imshow(pc.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(pc.shape[0]):
    for j in range(pc.shape[1]):
        v = pc.values[i, j]
        ax.text(j, i, "0" if v == 0 else f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.55 else INK, fontweight="bold" if v == 0 else None)
        if v == 0:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#e34948", linewidth=1.5))
ax.set_xticks(range(len(ARMS)), ARMS)
ax.set_yticks(range(len(labels)), labels)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("clean test F1 per class")
cb.outline.set_visible(False)
title(fig, "Per class, clean: Beaches and Coastal wetlands hit F1 = 0 in 5 and 3 of 6 arms; "
           "no class is 0 in every arm",
      "red outline = F1 exactly 0; class order = src/prepare_data.py:CLASSES")
save(fig, "fig_per_class_clean.png", rect=(0, 0, 1, 0.92))

ph = pd.read_csv(T / "per_class_harshest.csv")
fig, axes = plt.subplots(1, len(DEGRADATIONS), figsize=(15, 6.6), sharey=True)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
for k, (ax, d) in enumerate(zip(axes, DEGRADATIONS)):
    g = ph[ph["degradation"] == d].pivot(index="class", columns="model", values="delta")
    g = g.loc[pc.index, ARMS]
    im = ax.imshow(g.values, cmap=DIV, norm=norm, aspect="auto")
    sev = ph[ph["degradation"] == d]["severity"].iloc[0]
    ax.set_title(f"{d} @ {sev:g}", loc="left", color=INK, fontsize=9)
    ax.set_xticks(range(len(ARMS)), ARMS, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.25, 0.01, 0.55]))
cb.set_label("F1(harshest) - F1(clean), per class")
cb.outline.set_visible(False)
fig.suptitle("Per class at the harshest pilot severity: under full cloud only late keeps urban and "
             "water classes; gate_spatial keeps urban / industrial but not water", x=0.01, ha="left",
             color=INK, fontsize=11)
fig.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.14, wspace=0.08)
fig.savefig(F / "fig_per_class_harshest.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------- 4 modality weights
mw = pd.read_csv(T / "modality_weights.csv")
fusion_w = ["late", "gate_global", "gate_spatial"]
degs = ["cloud", "speckle", "temporal"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, d in zip(axes, degs):
    for m in fusion_w:
        g = pd.concat([mw[(mw["model"] == m) & (mw["degradation"] == "none")],
                       mw[(mw["model"] == m) & (mw["degradation"] == d)]]).sort_values(
            "severity")
        ax.plot(g["severity"], g["w_optical"], color=C[m], linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=m,
                zorder=3)
        if d == "speckle" and m != "late":
            ax.annotate(f"{m} {g['w_optical'].iloc[-1]:.2f}",
                        (g["severity"].iloc[-1], g["w_optical"].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK_MUTED)
    ax.axhline(0.5, color=INK_MUTED, linewidth=0.8, zorder=1)
    sev = sorted(set(mw[mw["degradation"] == d]["severity"]) | {0.0})
    ax.set_xticks(sev)
    ax.set_xlim(min(sev), max(sev) * 1.45)
    ax.set_ylim(0, 1)
    ax.set_title(f"{d} (corrupts {'optical' if d == 'cloud' else 'SAR'})"
                 if d != "temporal" else "temporal (misaligns SAR)",
                 loc="left", color=INK, fontsize=9)
    ax.set_xlabel("severity" if d != "temporal" else "grid steps")
    style(ax)
axes[0].set_ylabel("mean optical weight (SAR = 1 - optical)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, loc="lower center", ncol=3, labelcolor=INK_MUTED)
title(fig, "Modality weights: neither gate moves weight off a fully clouded optical input; "
           "gate_spatial moves it onto speckled SAR",
      "test-set mean of the per-sample (gate_global) / per-token (gate_spatial) weight; "
      "late's weight is one learned constant. drop_* force 0/1 by design (not shown)")
save(fig, "fig_modality_weights.png", rect=(0, 0.07, 1, 0.86))

# ---------------------------------------------------------------- 5 training health
loss = pd.read_csv(T / "train_loss.csv")
tp = pd.read_csv(T / "throughput.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for m in ARMS:
    g = loss[loss["model"] == m].sort_values("step")
    axes[0].plot(g["step"], g["loss"], color=C[m], linewidth=1.5, label=m)
    g = tp[tp["model"] == m].sort_values("epoch")
    axes[1].plot(g["epoch"] + 1, g["samples_per_s"], color=C[m], linewidth=2, marker="o",
                 markersize=4, markeredgecolor=SURFACE, label=m)
    axes[2].plot(g["epoch"] + 1, g["data_wait_frac"], color=C[m], linewidth=2, marker="o",
                 markersize=4, markeredgecolor=SURFACE, label=m)
axes[2].axhline(0.3, color="#e34948", linewidth=1)
axes[2].text(15, 0.31, "pre-flight limit 0.3", ha="right", va="bottom", fontsize=7,
             color=INK_MUTED)
axes[0].set(xlabel="optimizer step (batch 64)", ylabel="train loss (BCE, heartbeat sample)",
            title="")
axes[1].set(xlabel="epoch", ylabel="train samples / s", ylim=(0, 500))
axes[2].set(xlabel="epoch", ylabel="data_wait_frac (share of step time)", ylim=(0, 1))
for ax, t in zip(axes, ["train loss", "throughput", "input wait"]):
    ax.set_title(t, loc="left", color=INK, fontsize=9)
    style(ax)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, loc="lower center", ncol=6, labelcolor=INK_MUTED)
title(fig, "Training: losses fall without divergence, but every arm spent ~45-60% of each "
           "step waiting for data, flat at ~400-450 samples/s",
      "train-phase heartbeats every ~30 s (loss of the current batch); per-epoch throughput "
      "events. Validation F1 per epoch is not logged")
save(fig, "fig_training_health.png", rect=(0, 0.07, 1, 0.84))

# ---------------------------------------------------------------- 6 cost
t1 = pd.read_csv(REPORT / "analysis" / "exp1" / "table1.csv")
cost = clean[["params", "flops", "latency_ms", "f1_macro"]].copy()
cost["AUDC_cloud"] = t1[t1["degradation"] == "cloud"].set_index("model")["AUDC_mean"]
cost["AUDC_rel_cloud"] = t1[t1["degradation"] == "cloud"].set_index("model")["AUDC_rel_mean"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
specs = [("params", 1e3, "parameters (thousands)", "f1_macro", "clean macro-F1"),
         ("flops", 1e6, "FLOPs per sample (millions)", "f1_macro", "clean macro-F1"),
         ("params", 1e3, "parameters (thousands)", "AUDC_rel_cloud", "AUDC_rel under cloud")]
for ax, (x, sc, xl, y, yl) in zip(axes, specs):
    for m in ARMS:
        ax.scatter(cost.loc[m, x] / sc, cost.loc[m, y], s=60, color=C[m],
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=m)
    for m in ARMS:
        dx, dy, ha = {"late": (-7, 8, "right"), "gate_spatial": (7, 6, "left"),
                      "gate_global": (7, -10, "left")}.get(m, (7, 0, "left"))
        ax.annotate(m, (cost.loc[m, x] / sc, cost.loc[m, y]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", va="center", fontsize=7, color=INK_MUTED)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    xs = cost[x] / sc
    ax.set_xlim(xs.min() * 0.85, xs.max() * 1.2)
    style(ax, grid="both")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, loc="lower center", ncol=6, labelcolor=INK_MUTED)
title(fig, "Cost: doubling parameters (two encoders) buys no clean F1 and does not order "
           "robustness under cloud",
      "params and FLOPs from the clean result rows; latency_ms per sample in "
      "tables/clean.csv")
save(fig, "fig_cost.png", rect=(0, 0.07, 1, 0.86))
print(sorted(p.name for p in F.glob("*.png")))
