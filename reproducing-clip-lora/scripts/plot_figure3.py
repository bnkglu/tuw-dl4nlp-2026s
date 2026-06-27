#!/usr/bin/env python3
"""Reproduce the paper's Figure 3 from our own results (clip_lora_fig3.csv).

Per dataset (ImageNet, StanfordCars, EuroSAT): three rank x attention-matrix
heatmaps (Vision / Text / Vision+Text) + a placement bar chart (Up/Bottom/All),
matching the paper's layout. Output: results/figures/figure3_reproduction.png
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "clip_lora_fig3.csv")
OUT_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [("imagenet", "(a) ImageNet"),
            ("stanford_cars", "(b) Stanford Cars"),
            ("eurosat", "(c) EuroSAT")]
ENCODERS = [("vision", "Vision\n(All)"), ("text", "Text\n(All)"), ("both", "Vision and Text\n(All)")]
RANKS = [1, 2, 4, 8, 16, 32]
PARAMS = ["k", "q", "v", "o", "q_v", "q_v_k", "q_v_k_o"]
PARAM_LABELS = [r"$W_k$", r"$W_q$", r"$W_v$", r"$W_o$",
                r"$W_qW_v$", r"$W_qW_vW_k$", r"$W_qW_vW_kW_o$"]


def load():
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["_acc"] = float((r["accuracy"] or "").strip().rstrip("."))
            except ValueError:
                continue
            rows.append(r)
    return rows


def heat(rows, ds, enc):
    m = np.full((len(PARAMS), len(RANKS)), np.nan)
    for r in rows:
        if r["dataset"] == ds and r["encoder"] == enc and r["position"] == "all":
            if r["params"] in PARAMS and int(r["rank"]) in RANKS:
                m[PARAMS.index(r["params"]), RANKS.index(int(r["rank"]))] = r["_acc"]
    return m


def bars(rows, ds):
    out = {}
    for r in rows:
        if (r["dataset"] == ds and r["encoder"] == "both"
                and r["params"] == "q_k_v" and r["rank"] == "2"):
            out[r["position"]] = r["_acc"]
    return [out.get("up"), out.get("bottom"), out.get("all")]


def main():
    rows = load()
    fig = plt.figure(figsize=(15, 11))
    outer = GridSpec(3, 1, hspace=0.42, figure=fig)

    for di, (ds, title) in enumerate(DATASETS):
        # per-dataset colour scale across all three heatmaps
        mats = [heat(rows, ds, e) for e, _ in ENCODERS]
        vmin = np.nanmin([np.nanmin(m) for m in mats])
        vmax = np.nanmax([np.nanmax(m) for m in mats])
        gs = outer[di].subgridspec(1, 5, width_ratios=[1, 1, 1, 0.07, 0.9], wspace=0.25)

        im = None
        for ei, (enc, elabel) in enumerate(ENCODERS):
            ax = fig.add_subplot(gs[ei])
            m = mats[ei]
            im = ax.imshow(m, cmap="Greens", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(elabel, fontsize=10)
            ax.set_xticks(range(len(RANKS)), RANKS, fontsize=8)
            ax.set_xlabel("Rank", fontsize=9)
            if ei == 0:
                ax.set_yticks(range(len(PARAMS)), PARAM_LABELS, fontsize=9)
            else:
                ax.set_yticks([])
            for i in range(len(PARAMS)):
                for j in range(len(RANKS)):
                    if not np.isnan(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=6.5)

        cax = fig.add_subplot(gs[3])
        fig.colorbar(im, cax=cax)
        cax.set_ylabel("Top-1 Accuracy [%]", fontsize=8)

        # placement bar chart
        axb = fig.add_subplot(gs[4])
        vals = bars(rows, ds)
        labels = ["Up", "Bottom", "All"]
        cmap = plt.get_cmap("Greens")
        norm = plt.Normalize(min(v for v in vals if v) - 1, max(v for v in vals if v))
        axb.bar(labels, vals, color=[cmap(norm(v)) for v in vals], edgecolor="grey", linewidth=0.5)
        for i, v in enumerate(vals):
            axb.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        axb.set_title("Vision and Text", fontsize=10)
        axb.set_ylim(min(v for v in vals if v) - 1.5, max(v for v in vals if v) + 1.0)
        axb.tick_params(labelsize=8)
        axb.spines[["top", "right"]].set_visible(False)

        fig.text(0.5, outer[di].get_position(fig).y0 - 0.015, title,
                 ha="center", fontsize=12, style="italic")

    out = os.path.join(OUT_DIR, "figure3_reproduction.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
