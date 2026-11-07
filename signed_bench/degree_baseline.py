"""
degree_baseline.py
==================
Degree centrality by hand: top-k nodes by weighted degree
(sum of incident |weight|), ties by node index.  One run per k.
"""
import time


def run_degree(model, k: int) -> dict:
    t0 = time.perf_counter()
    S = sorted(range(model.n), key=lambda i: -model.deg[i])[:k]
    fitness = float(model.sigma(S))          # in-sample expected spread
    return {"seeds": S, "fitness": fitness,
            "time_s": time.perf_counter() - t0}