import time

from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics
from abcvns_paper import make_context, abcnvs_search

G = load_graph("bitcoin_alpha", verbose=False)
metrics = get_metrics(G)
model = SignedDiffusion(G, 30, seed=7)
ho = SignedDiffusion(G, 300, seed=999_999)
st_mr = make_context(G, 0.05, metrics, mode="mr", delta=0.15)

t0 = time.perf_counter()
S, stats = abcnvs_search(st_mr, 10, seed=0, verbose=False)
wall = time.perf_counter() - t0
pos, neg, tot = ho.components(S)
print(f"mr-fixed k=10: wall={wall:.1f}s fit={stats['LIE_best']:.1f}")
print(f"  held-out: pos={pos:.1f} neg={neg:.1f} tot={tot:.1f}")