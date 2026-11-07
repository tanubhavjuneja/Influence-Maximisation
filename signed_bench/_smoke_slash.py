import time

from abc_proposed import ABCConfig, abc_optimize
from abcvns_paper import make_context, abcnvs_search
from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics, signed_pool

G = load_graph("slashdot", verbose=False)
metrics = get_metrics(G)
model = SignedDiffusion(G, 30, seed=7)
ho = SignedDiffusion(G, 300, seed=999_999)
pool = signed_pool(model, metrics["pr"], metrics["core"], 200, seed=11)
st_lie = make_context(G, 0.01, metrics)
st_mr = make_context(G, 0.01, metrics, mode="mr", delta=0.15)

# small ABC cell
t0 = time.perf_counter()
cfg = ABCConfig(colony_size=8, max_cycles=5, eval_budget=2000,
                random_state=600)
r = abc_optimize(model, 10, pool, cfg, name="ABC (proposed)")
print(f"ABC k=10 small: t={time.perf_counter()-t0:.1f}s fit={r.spread:.1f}")
pos, neg, tot = ho.components(r.seeds)
print(f"  held-out pos={pos:.1f} neg={neg:.1f} tot={tot:.1f}")

# small fixed ABCVNS restart
t0 = time.perf_counter()
S, stats = abcnvs_search(st_mr, 10, seed=0, verbose=False,
                         params=dict(gmax=20, NP=20))
print(f"ABCVNS-fixed k=10 small: t={time.perf_counter()-t0:.1f}s "
      f"fit={stats['LIE_best']:.1f}")
pos, neg, tot = ho.components(S)
print(f"  held-out pos={pos:.1f} neg={neg:.1f} tot={tot:.1f}")

# small paper ABCVNS restart
t0 = time.perf_counter()
S2, stats2 = abcnvs_search(st_lie, 10, seed=0, verbose=False,
                           params=dict(gmax=20, NP=20))
print(f"ABCVNS-paper k=10 small: t={time.perf_counter()-t0:.1f}s "
      f"fit={stats2['LIE_best']:.1f}")
pos, neg, tot = ho.components(S2)
print(f"  held-out pos={pos:.1f} neg={neg:.1f} tot={tot:.1f}")
print("ALL OK")