"""
scheduler.py
============
Overnight scheduler: waits for the Bitcoin graphs to finish, then runs
Slashdot in the requested order:

  S1  Random, Degree, ABC (proposed)      -- ABC first with 5 restarts
  S2  ABC (proposed)                      -- then up to the full 30
  S3  ABCVNS (paper)  (30 restarts)
  S4  ABCVNS (fixed)  (30 restarts)
  S5  plots

Every phase is `runner.py` with the resumable results.csv, so the whole
chain can be interrupted (even by a server restart) and re-launched:
`python scheduler.py` resumes from the stored phase.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from common import log

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"
STATE = HERE / "schedule_state.json"

BITCOIN_TARGET = 920          # 2 graphs x 5 k x 92 cells

PHASES = [
    ("S1-slashdot-baselines-abc5",
     ["--graphs", "slashdot",
      "--algs", "Random", "Degree", "ABC (proposed)",
      "--restarts", "ABC (proposed)=5"]),
    ("S2-slashdot-abc30",
     ["--graphs", "slashdot",
      "--algs", "ABC (proposed)",
      "--restarts", "ABC (proposed)=30"]),
    ("S3-slashdot-abcvns-paper",
     ["--graphs", "slashdot",
      "--algs", "ABCVNS (paper)"]),
    ("S4-slashdot-abcvns-fixed",
     ["--graphs", "slashdot",
      "--algs", "ABCVNS (fixed)"]),
]


def bitcoin_cells() -> int:
    import csv
    n = 0
    if not (HERE / "results.csv").exists():
        return 0
    with open(HERE / "results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["graph"] in ("bitcoin_alpha", "bitcoin_otc"):
                n += 1
    return n


def load_state() -> int:
    if STATE.exists():
        return json.loads(STATE.read_text()).get("phase", 0)
    return 0


def save_state(phase: int) -> None:
    STATE.write_text(json.dumps({"phase": phase, "t": time.time()}))


def run_phase(cmd: list[str]) -> int:
    log(f"[scheduler] phase -> {' '.join(cmd)}")
    r = subprocess.run([sys.executable, str(RUNNER)] + cmd, cwd=str(HERE))
    return r.returncode


def main():
    phase = load_state()

    if phase == 0:
        log(f"[scheduler] waiting for bitcoin cells to reach "
            f"{BITCOIN_TARGET} (current {bitcoin_cells()}) ...")
        waited = 0
        while bitcoin_cells() < BITCOIN_TARGET:
            time.sleep(60)
            waited += 60
            if waited % 1800 == 0:
                log(f"[scheduler] still waiting: {bitcoin_cells()}/"
                    f"{BITCOIN_TARGET} bitcoin cells")
        phase = 1
        save_state(phase)

    while phase <= len(PHASES):
        name, args = PHASES[phase - 1]
        rc = 1
        for attempt in range(1, 4):
            rc = run_phase(args)
            if rc == 0:
                break
            log(f"[scheduler] phase {name} exited {rc} (attempt "
                f"{attempt}/3); retrying in 30 s ...")
            time.sleep(30)
        if rc != 0:
            log(f"[scheduler] phase {name} failed 3x; stopping "
                f"(re-launch scheduler.py to resume)")
            return
        phase += 1
        save_state(phase)

    # final plots
    for plot in ("plot_spread.py", "plot_time.py"):
        subprocess.run([sys.executable, str(HERE / plot)], cwd=str(HERE))
    save_state(99)
    log("[scheduler] all phases complete")


if __name__ == "__main__":
    main()