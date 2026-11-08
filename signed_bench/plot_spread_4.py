"""
plot_spread_4.py
================
Line chart: x = seed-set size k, y = held-out TOTAL spread — ONLY the four
required algorithms: Random, Degree, ABC (proposed), ABCVNS (paper).
One subplot per graph; values are means over restarts.  Reads results.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "results.csv"
ORDER = ["Random", "Degree", "ABC (proposed)", "ABCVNS (paper)"]
COLOURS = {"Random": "#e74c3c", "Degree": "#3498db",
           "ABC (proposed)": "#2ecc71", "ABCVNS (paper)": "#9b59b6"}
NAMES = {"Random": "Random", "Degree": "Degree",
         "ABC (proposed)": "ABC (proposed)",
         "ABCVNS (paper)": "ABCVNS (paper)"}


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["algorithm"] not in ORDER:
                continue
            data[row["graph"]][row["algorithm"]].append(
                (int(row["k"]), float(row["total_spread"])))
    return data


def main():
    data = load()
    graphs = sorted(data)
    fig, axes = plt.subplots(1, len(graphs), figsize=(6.0 * len(graphs), 4.6),
                             squeeze=False)
    for ax, g in zip(axes[0], graphs):
        for alg in ORDER:
            pts = sorted(data[g].get(alg, []))
            if not pts:
                continue
            ks = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            ax.plot(ks, vals, marker="o", label=alg, color=COLOURS[alg],
                    linewidth=2.2)
        ax.set_xlabel("seed set size k")
        ax.set_ylabel("held-out total spread (mean over restarts)")
        ax.set_title(g.replace("bitcoin_", "Bitcoin ").title())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Signed majority rule — spread: Random vs Degree vs ABC vs ABCVNS",
                 fontsize=13)
    fig.tight_layout()
    out = HERE / "plot_spread_4.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()