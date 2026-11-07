"""
calibrate.py
============
One-minute benchmark of the two hot operations on the ACTUAL machine,
so the projected runtimes are measured, not guessed:

  * one signed-cascade evaluation at R = 30   (drives ABC's cost),
  * one LIE evaluation                        (drives ABCVNS's cost).

Prints per-evaluation costs and the projected totals for the configured
restart counts (ABC: 5, ABCVNS: 30) over k = 10..50 for every enabled graph.
"""
import time

import numpy as np

from abcvns_paper import make_context
from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics
from runner import GRAPHS, KS, P_IC, POOL_SIZE

EVALS_PER_ABC = {10: 2000, 20: 2630, 30: 2630, 40: 2630, 50: 2630}
ABCVNS_EVALS = 25_000          # typical LIE calls per restart (search + VNS)
ABC_RESTARTS, ABCVNS_RESTARTS = 5, 30
R_FIT, R_HO = 30, 300


def main():
    print("== calibration (measured on this machine) ==")
    for name in GRAPHS:
        G = load_graph(name)
        metrics = get_metrics(G)
        model = SignedDiffusion(G, R_FIT, seed=7)
        st = make_context(G, P_IC[name], metrics)
        rng = np.random.default_rng(0)
        sets = [rng.choice(G["n"], 10, replace=False).tolist() for _ in range(20)]

        # --- one cascade evaluation ---
        model.sigma(sets[0])                     # warm up
        t0 = time.perf_counter()
        for S in sets:
            model.sigma(S)
        t_sigma = (time.perf_counter() - t0) / len(sets)

        # --- one LIE evaluation ---
        st.fitness(sets[0])
        t0 = time.perf_counter()
        for S in sets * 15:
            st.fitness(S)
        t_lie = (time.perf_counter() - t0) / (len(sets) * 15)

        abc_t = sum(EVALS_PER_ABC[k] * t_sigma for k in KS) * ABC_RESTARTS
        vns_t = ABCVNS_EVALS * t_lie * ABCVNS_RESTARTS * len(KS)
        print(f"\n{name}: sigma={t_sigma*1e3:7.1f} ms/eval  "
              f"LIE={t_lie*1e3:6.1f} ms/eval")
        print(f"  ABC   ({ABC_RESTARTS} restarts x {KS}): "
              f"{abc_t/3600:6.1f} h  (+ held-out {R_HO/R_FIT*t_sigma*ABC_RESTARTS*len(KS)/3600:.2f} h)")
        print(f"  ABCVNS ({ABCVNS_RESTARTS} restarts x {KS}): {vns_t/3600:6.1f} h")


if __name__ == "__main__":
    main()