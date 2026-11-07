"""
common.py
=========
Shared helpers for the signed-network benchmark:

  * results.csv cell store: append-per-cell with skip-on-re-run
    (restart-safe: an interrupted run resumes with zero recomputation),
  * timestamped logging to console AND to run.log,
  * the canonical CSV schema.
"""
import csv
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
LOG = HERE / "run.log"

# one row per finished (graph, k, algorithm, restart) cell
COLS = ["graph", "k", "algorithm", "restart", "time_s",
        "positive_spread", "negative_spread", "total_spread", "fitness"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_done() -> set[tuple]:
    """Keys of cells already written to results.csv."""
    done = set()
    if RESULTS.exists():
        with open(RESULTS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["graph"], int(row["k"]),
                          row["algorithm"], int(row["restart"])))
    return done


def append_cell(row: dict) -> None:
    """Append one finished cell (writes the header on first use)."""
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})