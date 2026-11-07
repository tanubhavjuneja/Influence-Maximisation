r"""
repair_real_csv.py
==================
One-off repair for the ragged CSVs written by the first version of
run_real_networks.py (append-per-cell with per-algorithm extra columns).

Problem: the header was written from the first (12-field) row, and later
rows carry extra x_* fields (CELF lazy counters, SA temperature, ABC
cycle stats, ABCVNS LIE stats), so pandas cannot read the file.

Every value is comma-free, so the repair is deterministic: split each
line on commas, map the trailing fields to the known per-algorithm extra
schema, and rewrite a clean CSV with a canonical header (all extra
columns present, empty where absent).

Usage: python repair_real_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = ["stage", "graph", "n", "m", "k", "algorithm", "restart", "spread",
        "spread_heldout", "runtime_s", "evaluations", "budget"]
EXTRA_ORDER = [
    "x_lazy_bound_checks", "x_lazy_bound_violations", "x_lazy_violation_rate",
    "x_temp_start", "x_restart",
    "x_cycles_run", "x_scout_resets", "x_dup_replacements",
    "x_refine_swaps", "x_refine_pair_swaps",
    "x_celf_seed_spread", "x_lie_best", "x_abcnvs_wall_s",
]
CANON = BASE + EXTRA_ORDER

# trailing extra fields per algorithm, IN THE ORDER THEY WERE WRITTEN
ALG_EXTRA = {
    "CELF": ["x_lazy_bound_checks", "x_lazy_bound_violations",
             "x_lazy_violation_rate"],
    "SA": ["x_temp_start", "x_restart"],
    "SimulatedAnnealing": ["x_temp_start", "x_restart"],
    "ABC (proposed)": ["x_cycles_run", "x_scout_resets", "x_dup_replacements",
                       "x_refine_swaps", "x_refine_pair_swaps",
                       "x_lazy_bound_checks", "x_lazy_bound_violations",
                       "x_celf_seed_spread"],
    "ABCVNS (paper)": ["x_lie_best", "x_abcnvs_wall_s"],
}


def repair(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in raw:
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if not fields or fields[0] == "stage":
            continue
        alg = fields[5]
        extra = ALG_EXTRA.get(alg, [])
        n_base, n_extra = len(BASE), len(extra)
        if len(fields) != n_base + n_extra:
            print(f"  [warn] {path.name}: {alg} row has {len(fields)} fields, "
                  f"expected {n_base + n_extra} -> skipping", file=sys.stderr)
            continue
        row = dict(zip(BASE, fields[:n_base]))
        row.update(dict(zip(extra, fields[n_base:])))
        rows.append(row)
    return rows


def write_clean(path: Path, rows: list[dict]):
    lines = [",".join(CANON)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in CANON))
    bak = path.with_suffix(".bak.csv")
    if bak.exists():
        bak.unlink()
    path.rename(bak)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  repaired {path.name}: {len(rows)} rows -> clean CSV "
          f"(backup: {bak.name})")


def main():
    results = Path("results")
    for f in sorted(results.glob("real_*.csv")):
        rows = repair(f)
        if rows:
            write_clean(f, rows)
        else:
            print(f"  {f.name}: nothing to repair")


if __name__ == "__main__":
    main()