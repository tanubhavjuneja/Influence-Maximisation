"""
calibrate_opinion.py
====================
Measure the per-evaluation cost of the two workloads on the CURRENT backend
(CPU or GPU, chosen by backend.py / SB_BACKEND):

  * sigma  - one held-out diffusion evaluation of the two-opinion model (R=30)
  * LIE    - one majority-rule surrogate fitness evaluation (ABCVNS fixed)

Run twice (SB_BACKEND=cpu and GPU) and take the ratio to convert the
CPU-measured benchmark cells to GPU-equivalent estimates.
"""
import json
import sys
import time

import numpy as np

import backend as be
from abcvns_paper import make_context
from graphio import load_graph
from metrics import get_metrics
from opinion_model import SignedOpinionModel

P_IC = {"bitcoin_alpha": 0.05, "bitcoin_otc": 0.05, "slashdot": 0.01}
GRAPHS = sys.argv[1:] or ["bitcoin_alpha", "bitcoin_otc", "slashdot"]

out = {}
for name in GRAPHS:
    G = load_graph(name, verbose=False)
    m = get_metrics(G)
    model = SignedOpinionModel(G, 30, seed=7, comm=m["comm"])
    rng = np.random.default_rng(0)
    sets = [rng.choice(G["n"], 10, replace=False).tolist() for _ in range(6)]

    model.sigma_pos(sets[0])                      # warm-up
    t0 = time.perf_counter()
    for S in sets[1:]:
        model.sigma_pos(S)
    sigma_ms = (time.perf_counter() - t0) / 5 * 1e3

    st = make_context(G, P_IC[name], m, mode="mr")
    for S in sets:
        st.fitness(S)                             # warm-up
    reps = 5 if name == "slashdot" else 20
    t0 = time.perf_counter()
    for _ in range(reps):
        for S in sets:
            st.fitness(S)
    lie_ms = (time.perf_counter() - t0) / (reps * len(sets)) * 1e3

    out[name] = {"sigma_ms": round(sigma_ms, 4),
                 "lie_ms": round(lie_ms, 4),
                 "backend": "gpu" if be.GPU else "cpu"}
    print(f"{name}: {out[name]}", flush=True)

print("JSON " + json.dumps(out))
