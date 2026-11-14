"""
plot_opinion.py
===============
Charts for the corrected two-opinion signed majority-rule benchmark.

  opinion_spread.png   net spread (positive - negative) vs k
  opinion_quality.png  diversity (top) and community fairness (bottom) vs k
  opinion_time.png     wall-clock vs k
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "results_opinion.csv"
ORDER = ["ABC (proposed)", "ABC (plain)", "ABCVNS (fixed)",
         "SignedPageRank", "Degree"]
COLOURS = {"ABC (proposed)": "#2ecc71", "ABC (plain)": "#95a5a6",
           "ABCVNS (fixed)": "#e67e22", "SignedPageRank": "#9b59b6",
           "Degree": "#3498db"}


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["graph"]][row["algorithm"]].append(row)
    return data


def series(rows, k, col):
    vals = [float(r[col]) for r in rows if int(r["k"]) == k]
    return sum(vals) / len(vals) if vals else None


def panels(data, metric, fname, title, ylabel):
    graphs = sorted(data)
    fig, axes = plt.subplots(1, len(graphs), figsize=(6.0 * len(graphs), 4.6),
                             squeeze=False)
    for ax, g in zip(axes[0], graphs):
        for alg in ORDER:
            rows = data[g].get(alg, [])
            if not rows:
                continue
            ks = sorted({int(r["k"]) for r in rows})
            vals = [series(rows, k, metric) for k in ks]
            ax.plot(ks, vals, marker="o", label=alg, color=COLOURS[alg],
                    linewidth=2.2)
        ax.set_xlabel("seed set size k")
        ax.set_ylabel(ylabel)
        ax.set_title(g.replace("bitcoin_", "Bitcoin ").title())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=150)
    print(f"wrote {HERE / fname}")


def main():
    data = load()
    panels(data, "net_spread", "opinion_spread.png",
           "Two-opinion signed majority rule — net spread (positive − negative)",
           "net spread (mean over restarts)")
    panels(data, "time_s", "opinion_time.png",
           "Two-opinion signed majority rule — wall-clock",
           "seconds (mean over restarts)")
    panels(data, "diversity", "opinion_diversity.png",
           "Two-opinion signed majority rule — opinion diversity",
           "normalised entropy (0 = one opinion, 1 = 50/50)")
    panels(data, "fairness", "opinion_fairness.png",
           "Two-opinion signed majority rule — community fairness",
           "1 − (max − min) community activation rate")


if __name__ == "__main__":
    main()