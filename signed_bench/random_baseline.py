"""
random_baseline.py
==================
Random seed selection: k uniformly random nodes.  Deterministic per k
(fixed seed), one run per k.
"""
import time

import numpy as np


def run_random(model, k: int, seed: int = 12_345) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + k)
    S = rng.choice(model.n, size=k, replace=False).tolist()
    fitness = float(model.sigma(S))          # in-sample expected spread
    return {"seeds": S, "fitness": fitness,
            "time_s": time.perf_counter() - t0}