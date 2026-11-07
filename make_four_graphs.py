"""
make_four_graphs.py
===================
Build the four report figures directly from results/real_*.csv:

  1. results/fig_real_bitcoin_spread.png   spread, Bitcoin Alpha + OTC
  2. results/fig_real_bitcoin_time.png     wall-clock, Bitcoin Alpha + OTC
  3. results/fig_real_slashdot_spread.png  spread, Slashdot (majority + IC)
  4. results/fig_real_slashdot_time.png    wall-clock, Slashdot (majority + IC)

Bitcoin figures have one row per diffusion model (majority, IC); Slashdot
figures have one panel per diffusion model.  Bars are grouped by k and every
panel is sorted by mean spread so the best algorithm is at the top.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
COLORS = plt.get_cmap("viridis")


def load(stage: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"real_{stage}.csv")


def draw_panel(ax, df: pd.DataFrame, graph: str, value: str,
               title: str, logx: bool = False):
    sub = df[df.graph == graph]
    ks = sorted(sub.k.unique())
    order = (sub.groupby("algorithm")["spread_heldout"].mean()
             .sort_values(ascending=True).index.tolist())
    y = np.arange(len(order))
    h = 0.8 / max(1, len(ks))
    for i, k in enumerate(ks):
        vals = (sub[sub.k == k].groupby("algorithm")[value].mean()
                .reindex(order).fillna(0.0).astype(float))
        if logx:
            vals = vals.clip(lower=0.01)
        ax.barh(y + (i - (len(ks) - 1) / 2) * h, vals.values,
                height=h, color=COLORS(0.15 + 0.7 * i / max(1, len(ks) - 1)),
                label=f"k={k}")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    if logx:
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock per configuration (s, log scale)", fontsize=8)
    else:
        ax.set_xlabel("held-out spread", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)


def finish(fig, path: Path, note: str = ""):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def bitcoin(df_maj, df_ic):
    for value, logx, tag, ylab in [
        ("spread_heldout", False, "spread", "held-out spread"),
        ("runtime_s", True, "time", "wall-clock (s)"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        for row, (df, model) in enumerate([(df_maj, "majority"),
                                           (df_ic, "IC")]):
            for col, graph in enumerate(["bitcoin_alpha", "bitcoin_otc"]):
                draw_panel(axes[row, col], df, graph, value,
                           f"{graph} — {model}", logx=logx)
        fig.suptitle(f"Bitcoin networks: {ylab}", fontsize=13, y=0.995)
        note = ""
        if value == "spread_heldout":
            note = ("majority: ABCVNS bars are 10.2 / 20.2 / 31.5 (Alpha) and "
                    "10.1 / 22.8 / 32.4 (OTC) — LIE transfer failure; "
                    "IC: p=0.05 saturation makes all methods ~equal")
        else:
            note = "bars at the 0.01 s floor are sub-10 ms heuristics"
        finish(fig, RESULTS / f"fig_real_bitcoin_{tag}.png", note)


def slashdot(df_maj, df_ic):
    for value, logx, tag, ylab in [
        ("spread_heldout", False, "spread", "held-out spread"),
        ("runtime_s", True, "time", "wall-clock (s)"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        draw_panel(axes[0], df_maj, "slashdot", value,
                   "Slashdot — majority rule", logx=logx)
        draw_panel(axes[1], df_ic, "slashdot", value,
                   "Slashdot — independent cascade (p=0.01)", logx=logx)
        fig.suptitle(f"Slashdot (82,140 nodes): {ylab}", fontsize=13, y=0.99)
        note = ("majority: SignedPR best (1,827 / 2,293); "
                "IC: ABCVNS and ABC within 0.3%" if value == "spread_heldout"
                else "log scale; bars at 0.01 s are sub-10 ms heuristics")
        finish(fig, RESULTS / f"fig_real_slashdot_{tag}.png", note)


def main():
    df_maj = load("majority")
    df_ic = load("ic")
    bitcoin(df_maj, df_ic)
    slashdot(df_maj, df_ic)


if __name__ == "__main__":
    main()
