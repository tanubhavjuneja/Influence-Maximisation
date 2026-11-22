"""Why is the optimised ABC slower per evaluation on Slashdot?
Compare per-eval time and diffusion rounds for random vs hub seed sets."""
import sys
import time

import numpy as np

sys.path.insert(0, "signed_bench")

from graphio import load_graph
from metrics import get_metrics
from opinion_model import SignedOpinionModel

G = load_graph("slashdot", verbose=False)
met = get_metrics(G)
model = SignedOpinionModel(G, 30, seed=7, comm=met["comm"])
n = model.n

# standalone CPU cascade with a round counter (same rule as the model)
def run_count(S):
    R = model.R
    active = np.zeros((n, R), dtype=np.int8)
    active[np.asarray(S, dtype=np.int64), :] = 1
    support = np.zeros((n, R), dtype=np.float32)
    frontier = active.copy()
    rr = np.arange(R, dtype=np.int64)
    rounds = 0
    while frontier.any():
        rounds += 1
        e = frontier[model.src].any(axis=1)
        if not e.any():
            break
        s_sel, d_sel, w_sel = model.src[e], model.dst[e], model.sw[e]
        fr = frontier[s_sel].astype(np.float32)
        for c0 in range(0, int(s_sel.shape[0]), 40000):
            d_ = d_sel[c0:c0 + 40000]
            w_ = w_sel[c0:c0 + 40000]
            f_ = fr[c0:c0 + 40000]
            idx = d_[:, None] * R + rr[None, :]
            np.add.at(support.ravel(), idx.ravel(), (w_[:, None] * f_).ravel())
        newly = (np.abs(support) >= model.need_np) & (active == 0)
        if not newly.any():
            break
        sgn = (support > 0).astype(np.int8) - (support < 0).astype(np.int8)
        active = np.where(newly, sgn, active)
        frontier = newly
    return int((active == 1).sum(axis=0).mean()), rounds

rng = np.random.default_rng(0)
random_sets = [rng.choice(n, 10, replace=False).tolist() for _ in range(20)]
hub = np.argsort(-model.deg)[:10].tolist()
net = np.argsort(-(model.posw - model.negw))[:10].tolist()

for name, sets in [("random", random_sets),
                   ("hub", [hub] * 20),
                   ("net-positive", [net] * 20)]:
    for S in sets[:1]:
        run_count(S)                      # warm-up
    t0 = time.perf_counter()
    out = [run_count(S) for S in sets]
    dt = (time.perf_counter() - t0) / len(sets)
    pos = np.mean([o[0] for o in out])
    rounds = np.mean([o[1] for o in out])
    print(f"{name:13s} {dt*1e3:7.1f} ms/eval   pos={pos:7.1f}   rounds={rounds:4.1f}")
