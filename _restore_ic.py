"""Restore the four Slashdot IC ABCVNS rows purged by accident.

All values are the exact measured numbers from results/real_run.log and the
pre-purge raw dumps.  The k=10 in-sample `spread` values were never logged,
so they stay empty (held-out spread is the authoritative column).
"""
import pandas as pd
from run_real_networks import CANON_COLS

rows = [
    # k, restart, spread, spread_heldout, runtime_s, lie_best, abcnvs_wall_s
    (10, 0, None, 1492.138, 260.70, 269.925, 260.3),
    (10, 1, None, 1486.261, 259.83, 271.440, 259.4),
    (20, 0, 1592.280029, 1586.546997, 413.525541, 351.599, 413.2),
    (20, 1, 1584.829956, 1588.359009, 458.489241, 351.444, 458.2),
]

df = pd.read_csv("results/real_ic.csv")
existing = set(zip(df.graph, df.k, df.algorithm, df.restart))
added = 0
for k, rs, sp, ho, t, lie, wall in rows:
    if ("slashdot", k, "ABCVNS (paper)", rs) in existing:
        print(f"already present: k={k} rs={rs}; skipping")
        continue
    row = {"stage": "ic", "graph": "slashdot", "n": 82140, "m": 500481,
           "k": k, "algorithm": "ABCVNS (paper)", "restart": rs,
           "spread": "" if sp is None else sp, "spread_heldout": ho,
           "runtime_s": t, "evaluations": 1, "budget": k * 200,
           "x_lie_best": lie, "x_abcnvs_wall_s": wall}
    line = ",".join(str(row.get(c, "")) for c in CANON_COLS)
    with open("results/real_ic.csv", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    added += 1

print(f"appended {added} restored rows")

d = pd.read_csv("results/real_ic.csv")
a = d[d.algorithm == "ABCVNS (paper)"]
print("total ic rows:", len(d), "| abcnvs rows:", len(a))
print(a.groupby(["graph", "k"])["spread_heldout"]
      .agg(["count", "mean"]).round(1).to_string())