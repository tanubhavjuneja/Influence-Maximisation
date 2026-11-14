"""
retime_opinion.py
=================
Rewrite results_opinion.csv so every timing is on the SAME basis (GPU).

The original run mixed backends: all Bitcoin cells and Slashdot k=10 were
measured on numpy CPU; Slashdot k=20/30 on the GPU.  That makes the time
columns incomparable (e.g. Slashdot ABC k=10 = 419.7 s on CPU vs 54.1 s at
k=20 on GPU).

Conversion ratios come from GPU anchor runs of the exact same configurations
(anchor_opinion.py) plus per-workload microbenchmarks (calibrate_opinion.py):

  ratio = time_CPU / time_GPU   (divide the CPU measurement by it)

  bitcoin_alpha  ABC*            1.0656   (13.0 s -> 12.2 s)
  bitcoin_alpha  ABCVNS (fixed)  0.2387   ( 9.5 s -> 39.8 s; LIE is
                                           launch-overhead-bound on GPU)
  bitcoin_otc    ABC*            1.873    (23.6 s -> 12.6 s)
  bitcoin_otc    ABCVNS (fixed)  0.2817   (11.1 s -> 39.4 s)
  slashdot       ABC (proposed)  8.677    (420.8 s -> 48.5 s)
  slashdot       ABC (plain)    14.231    (436.9 s -> 30.7 s)
  slashdot       ABCVNS (fixed)  8.60    (566.7 s -> 65.9 s; from the
                                           dual-backend cells rs0-2 vs rs3-4)

Already-GPU cells (Slashdot k=20/30, Slashdot k=10 ABCVNS rs3-4) are kept
unchanged.  New columns:
  time_raw_s  - the value as originally measured
  time_basis  - measured_gpu | estimated_gpu | structural (baselines)
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "results_opinion.csv"

RATIOS = {
    ("bitcoin_alpha", "ABC (proposed)"): 13.0 / 12.2,
    ("bitcoin_alpha", "ABC (plain)"): 13.0 / 12.2,
    ("bitcoin_alpha", "ABCVNS (fixed)"): 9.5 / 39.8,
    ("bitcoin_otc", "ABC (proposed)"): 23.6 / 12.6,
    ("bitcoin_otc", "ABC (plain)"): 23.6 / 12.6,
    ("bitcoin_otc", "ABCVNS (fixed)"): 11.1 / 39.4,
    ("slashdot", "ABC (proposed)"): 420.8 / 48.5,
    ("slashdot", "ABC (plain)"): 436.9 / 30.7,
    ("slashdot", "ABCVNS (fixed)"): 566.7 / 65.9,
}
BASELINES = {"SignedPageRank", "Degree"}


def is_measured_gpu(row):
    if row["graph"] != "slashdot":
        return False
    if int(row["k"]) >= 20:
        return True
    return row["algorithm"] == "ABCVNS (fixed)" and int(row["restart"]) >= 3


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)

    for row in rows:
        raw = float(row["time_s"])
        row["time_raw_s"] = f"{raw:.4f}"
        if row["algorithm"] in BASELINES:
            row["time_s"] = f"{raw:.4f}"
            row["time_basis"] = "structural"
        elif is_measured_gpu(row):
            row["time_s"] = f"{raw:.4f}"
            row["time_basis"] = "measured_gpu"
        else:
            ratio = RATIOS[(row["graph"], row["algorithm"])]
            row["time_s"] = f"{raw / ratio:.4f}"
            row["time_basis"] = "estimated_gpu"

    out_cols = cols + ["time_raw_s", "time_basis"]
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"rewrote {CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
