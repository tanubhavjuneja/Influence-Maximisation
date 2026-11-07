import time

import numpy as np

import backend as be
from abcvns_paper import make_context
from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics

print("backend:", be.BACKEND, "GPU:", be.GPU)
G = load_graph("slashdot", verbose=False)
metrics = get_metrics(G)

model = SignedDiffusion(G, 30, seed=7)
rng = np.random.default_rng(0)
sets = [rng.choice(G["n"], 10, replace=False).tolist() for _ in range(10)]

model.sigma(sets[0])
t0 = time.perf_counter()
for S in sets:
    model.sigma(S)
t_sig = (time.perf_counter() - t0) / len(sets)

st = make_context(G, 0.01, metrics)
st.fitness(sets[0])
t0 = time.perf_counter()
for S in sets * 20:
    st.fitness(S)
t_lie = (time.perf_counter() - t0) / (len(sets) * 20)

print(f"sigma = {t_sig*1e3:7.1f} ms/eval")
print(f"LIE   = {t_lie*1e3:6.1f} ms/eval")

ABC_EVALS = {10: 2000, 20: 2630, 30: 2630, 40: 2630, 50: 2630}
VNS_EVALS = 25000
abc5 = sum(ABC_EVALS[k] * t_sig for k in [10, 20, 30, 40, 50]) * 5 / 3600
abc25 = sum(ABC_EVALS[k] * t_sig for k in [10, 20, 30, 40, 50]) * 25 / 3600
vns = VNS_EVALS * t_lie * 30 * 5 / 3600
print(f"\nprojected slashdot (R_fit=30, k=10..50):")
print(f"  ABC 5 restarts        : {abc5:6.1f} h")
print(f"  ABC 25 more restarts  : {abc25:6.1f} h")
print(f"  ABCVNS paper 30 rs    : {vns:6.1f} h")
print(f"  ABCVNS fixed 30 rs    : {vns:6.1f} h")
print(f"  baselines             : ~0.1 h")
print(f"  TOTAL                 : {abc5+abc25+2*vns+0.1:6.1f} h")