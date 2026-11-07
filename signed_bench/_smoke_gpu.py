import time

import numpy as np

import backend as be
from abcvns_paper import make_context, lie_fitness
from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics

print("GPU:", be.GPU, "backend:", be.BACKEND)

G = load_graph("bitcoin_alpha", verbose=False)
metrics = get_metrics(G)
model = SignedDiffusion(G, 30, seed=7)

# sanity: degree top-10 spread should be ~439 (CPU reference)
S = sorted(range(model.n), key=lambda i: -model.deg[i])[:10]
pos, neg, tot = model.components(S)
print(f"degree k=10: pos={pos:.1f} neg={neg:.1f} tot={tot:.1f} (expect ~439)")

# LIE sanity
st = make_context(G, 0.05, metrics)
print(f"LIE(deg10) = {lie_fitness(st, S):.1f} (expect ~800-1500)")

# timing
rng = np.random.default_rng(0)
sets = [rng.choice(G["n"], 10, replace=False).tolist() for _ in range(20)]
model.sigma(sets[0])
t0 = time.perf_counter()
for s in sets:
    model.sigma(s)
print(f"sigma: {(time.perf_counter()-t0)/len(sets)*1e3:.1f} ms/eval")

st.fitness(sets[0])
t0 = time.perf_counter()
for s in sets * 15:
    st.fitness(s)
print(f"LIE:   {(time.perf_counter()-t0)/(len(sets)*15)*1e3:.2f} ms/eval")