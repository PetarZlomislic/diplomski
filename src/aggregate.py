"""Aggregation entrypoint: results dir -> tidy dataframe + degradation-curve plots.

Each run writes its own results.csv (no cross-machine file locking); this merges them.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.results import FIELDNAMES  # noqa: E402

# Validated categorical palette (light surface), assigned in fixed order, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_results(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("results.csv"))
    if not files:
        raise FileNotFoundError(f"no results.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)[FIELDNAMES]
    # Re-evaluating the same checkpoint supersedes earlier rows instead of double-counting.
    key = ["run_id", "degradation", "severity", "ckpt_kind", "threshold"]
    df = df.sort_values("timestamp").drop_duplicates(key, keep="last")
    return df.sort_values(["model", "degradation", "severity"], ignore_index=True)


def model_colors(models: list[str]) -> dict[str, str]:
    """Color follows the model name (sorted), not its rank in a given figure."""
    names = sorted(set(models))
    if len(names) > len(SERIES_COLORS):
        raise ValueError(f"{len(names)} models > {len(SERIES_COLORS)} colors: facet instead")
    return dict(zip(names, SERIES_COLORS))


def _spread_labels(ends: list[list], min_gap: float) -> list[list]:
    """Nudge end-label y positions apart so converging lines keep readable labels."""
    ends = sorted(ends, key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    return ends


def plot_curve(df: pd.DataFrame, degradation: str, metric: str, path: Path,
               colors: dict[str, str]) -> None:
    d = df[df["degradation"] == degradation]
    curve = d.groupby(["model", "severity"], as_index=False)[metric].mean()

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for model, g in curve.groupby("model"):
        g = g.sort_values("severity")
        ax.plot(g["severity"], g[metric], color=colors[model], linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=model,
                zorder=3, clip_on=False)
        ends.append([g[metric].iloc[-1], model, g["severity"].iloc[-1]])
    for y, model, x in _spread_labels(ends, min_gap=0.05):
        ax.annotate(model, (x, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_MUTED, annotation_clip=False)

    ax.set_xlabel("severity", color=INK_MUTED)
    ax.set_ylabel(metric, color=INK_MUTED)
    ax.set_title(f"{metric} under {degradation}", color=INK, loc="left", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xlim(curve["severity"].min(), curve["severity"].max() * 1.12 or 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=min(4, len(colors)), borderaxespad=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def aggregate(root: Path, out: Path, metric: str = "f1_macro") -> tuple[pd.DataFrame, list[Path]]:
    df = load_results(root)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_all.csv", index=False)
    colors = model_colors(df["model"].tolist())
    plots = []
    for deg in sorted(df["degradation"].unique()):
        if deg == "none":
            continue
        p = out / f"curve_{deg}_{metric}.png"
        plot_curve(df, deg, metric, p, colors)
        plots.append(p)
    return df, plots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory searched recursively for results.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/aggregate")
    ap.add_argument("--metric", default="f1_macro", choices=["f1_macro", "f1_micro", "accuracy"])
    args = ap.parse_args()
    df, plots = aggregate(args.root, args.out or args.root / "aggregate", args.metric)
    print(f"{len(df)} rows; plots: {', '.join(str(p) for p in plots) or 'none'}")


if __name__ == "__main__":
    main()
