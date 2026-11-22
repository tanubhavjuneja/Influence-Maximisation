"""Measure how many cascade evaluations the CELF warm start consumes."""
import sys
import time

sys.path.insert(0, "signed_bench")

from abc_proposed import celf
from graphio import load_graph
from metrics import get_metrics, signed_pool
from opinion_model import SignedOpinionModel

for name in ["bitcoin_alpha", "bitcoin_otc", "slashdot"]:
    G = load_graph(name, verbose=False)
    met = get_metrics(G)
    model = SignedOpinionModel(G, 30, seed=7, comm=met["comm"])
    pool = signed_pool(model, met["pr"], met["core"], 200, seed=11)
    for k in [10, 20, 30]:
        model.reset_counters()
        model.clear_cache()
        t0 = time.perf_counter()
        r = celf(model, k, pool=pool)
        dt = time.perf_counter() - t0
        print(f"{name:14s} k={k:2d}  celf_evals={r.evaluations:5d}  "
              f"budget={k*200:5d}  left={k*200-r.evaluations:5d}  "
              f"time={dt:6.1f}s", flush=True)
