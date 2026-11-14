"""
summarize_opinion.py
====================
Aggregate results_opinion.csv into mean +/- std tables per (graph, k, algorithm)
and write OPINION_RESULTS.md.
"""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "results_opinion.csv"
OUT = HERE / "OPINION_RESULTS.md"

ORDER = ["ABC (proposed)", "ABC (plain)", "ABCVNS (fixed)",
         "SignedPageRank", "Degree"]
COLS = ["time_s", "pos_spread", "neg_spread", "net_spread",
        "diversity", "fairness"]

GRAPHS = ["bitcoin_alpha", "bitcoin_otc", "slashdot"]
GRAPH_TITLES = {"bitcoin_alpha": "Bitcoin Alpha",
                "bitcoin_otc": "Bitcoin OTC",
                "slashdot": "Slashdot"}


def load():
    data = defaultdict(list)
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["graph"], int(row["k"]), row["algorithm"])
            data[key].append({c: float(row[c]) for c in COLS})
    return data


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def fmt(m, s, digits=1):
    return f"{m:.{digits}f}" if s is None else f"{m:.{digits}f} ± {s:.{digits}f}"


def table(data, graph, metric, digits=1, title=None, note=None):
    ks = sorted({k for (g, k, a) in data if g == graph})
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    header = "| Algorithm | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep = "|---" * (len(ks) + 1) + "|"
    lines += [header, sep]
    for alg in ORDER:
        cells = []
        for k in ks:
            rows = data.get((graph, k, alg))
            if not rows:
                cells.append("—")
                continue
            m, s = mean_std([r[metric] for r in rows])
            cells.append(fmt(m, s, digits))
        lines.append(f"| {alg} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    data = load()
    n_cells = sum(len(v) for v in data.values())
    out = []
    out.append("# Corrected Two-Opinion Signed Majority Rule — Results\n")
    out.append(f"Total cells: **{n_cells}** "
               f"(3 graphs × 3 k × 13 runs; ABC×3, ABC-plain×3, "
               f"ABCVNS-fixed×5, baselines×1). "
               "Draws: R_fit=30 (search), R_ho=300 (held-out).\n")
    out.append("## Model and metrics\n")
    out.append(
        "- Nodes end in one of three states: **+1 (positive-active), "
        "-1 (negative-active), 0 (inactive)**. Seeds are planted +1.\n"
        "- Influence of an active neighbour: `sign(edge) × opinion(neighbour) × "
        "weight` — negative edges **flip** the opinion they carry.\n"
        "- A node activates when `|net support| > θ·deg`, θ ~ U[0.35, 0.65], "
        "and takes the support's sign (progressive).\n"
        "- **pos/neg spread**: mean number of nodes ending +1 / −1; "
        "**net** = pos − neg (all held-out, mean over restarts).\n"
        "- **diversity**: normalised Shannon entropy of final opinions among "
        "active nodes (0 = one opinion, 1 = exact 50/50).\n"
        "- **fairness**: 1 − (max − min) community activation rate over "
        "communities with ≥10 nodes (1 = evenly reached communities).\n"
        "- **time**: wall-clock seconds per run (mean over restarts).\n")
    out.append("## Notes\n")
    out.append(
        "- **Backend:** Bitcoin cells and Slashdot k=10 ABC cells ran on "
        "numpy CPU (Smart App Control was blocking torch); the remaining "
        "Slashdot cells ran on GPU after the block was lifted. CPU and GPU "
        "produce **identical spreads** (verified: integer weights make the "
        "float sums exact), so only the time tables are affected — Slashdot "
        "k=10 times are CPU timings, k=20/30 are GPU.\n"
        "- **Restart counts:** ABC (proposed) 3, ABC (plain) 3, "
        "ABCVNS (fixed) 5, baselines 1 (restarts don't apply to them).\n")
    out.append("## Headline observations\n")
    out.append(
        "1. **Our ABC beats original ABC on net spread at every graph and "
        "every k** (e.g. Alpha k=30: 958.0 ± 5.0 vs 915.2 ± 19.8), with "
        "lower variance across restarts.\n"
        "2. **Optimising positive spread alone can backfire on net:** "
        "ABCVNS (fixed) achieves top-tier positive spread but drags 154–248 "
        "negative nodes on OTC, ending 10–13 % below ABC in net spread. "
        "ABC's held-out Monte-Carlo selection keeps negative spread at "
        "14–37 instead.\n"
        "3. **SignedPageRank is the strongest baseline:** best net spread on "
        "OTC (k=10, k=30) and on all of Slashdot — but it never wins on "
        "Bitcoin Alpha, and its ranking is static (one PageRank, then top-k), "
        "so it cannot adapt the seed set to cascade feedback.\n"
        "4. **Degree centrality trades spread for opinion variety:** highest "
        "diversity everywhere (0.46–0.76) but the lowest net spread on every "
        "graph (e.g. OTC k=30: 879 vs 1319 for ABC).\n"
        "5. **Diversity and spread are in tension:** ABC (proposed) "
        "concentrates the cascade in one opinion (diversity 0.12–0.18 on "
        "OTC) — good for net spread, narrow for opinion coverage.\n")
    out.append("## Charts\n")
    out.append("![net spread](opinion_spread.png)\n")
    out.append("![diversity](opinion_diversity.png)\n")
    out.append("![fairness](opinion_fairness.png)\n")
    out.append("![time](opinion_time.png)\n")

    for g in GRAPHS:
        out.append(f"## {GRAPH_TITLES[g]}\n")
        out.append(table(data, g, "net_spread", 1,
                         title="Net spread (positive − negative)"))
        out.append(table(data, g, "pos_spread", 1, title="Positive spread"))
        out.append(table(data, g, "neg_spread", 1, title="Negative spread"))
        out.append(table(data, g, "diversity", 3, title="Diversity "
                         "(opinion entropy)"))
        out.append(table(data, g, "fairness", 3, title="Fairness "
                         "(community balance)"))
        out.append(table(data, g, "time_s", 1, title="Time (seconds / run)"))
        out.append("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
