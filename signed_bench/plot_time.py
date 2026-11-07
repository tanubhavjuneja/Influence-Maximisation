"""
plot_time.py
============
Line chart #2: x = seed-set size k, y = wall-clock time (s),
one coloured line per algorithm, one subplot per graph.
Values are means over restarts.  Reads results.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "results.csv"
ORDER = ["Random", "Degree", "ABC (proposed)", "ABCVNS (paper)",
         "ABCVNS (fixed)"]
COLOURS = {"Random": "#e74c3c", "Degree": "#3498db",
           "ABC (proposed)": "#2ecc71", "ABCVNS (paper)": "#9b59b6",
           "ABCVNS (fixed)": "#e67e22"}


def load():
    data = defaultdict(lambda: defaultdict(list))   # graph -> alg -> (k, t)
    if not CSV.exists():
        raise SystemExit("results.csv not found; run runner.py first")
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["graph"]][row["algorithm"]].append(
                (int(row["k"]), float(row["time_s"])))
    return data


def main():
    data = load()
    graphs = sorted(data)
    fig, axes = plt.subplots(1, len(graphs), figsize=(6.5 * len(graphs), 5),
                             squeeze=False)
    for ax, g in zip(axes[0], graphs):
        for alg in ORDER:
            pts = sorted(data[g].get(alg, []))
            if not pts:
                continue
            ks = [p[0] for p in pts]
            ts = [p[1] for p in pts]
            ax.plot(ks, ts, marker="o", label=alg, color=COLOURS[alg],
                    linewidth=2)
        ax.set_xlabel("seed set size k")
        ax.set_ylabel("wall-clock time (s, mean over restarts)")
        ax.set_title(g)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Signed majority rule: wall-clock vs seed set size",
                 fontsize=13)
    fig.tight_layout()
    out = HERE / "plot_time.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()