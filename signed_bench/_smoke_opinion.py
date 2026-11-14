import time

from abc_proposed import ABCConfig, abc_optimize
from abcvns_paper import make_context, abcnvs_search
from graphio import load_graph
from metrics import get_metrics, signed_pool, signed_pagerank
from opinion_model import SignedOpinionModel

G = load_graph("bitcoin_alpha", verbose=False)
metrics = get_metrics(G)
metrics["spr"] = signed_pagerank(G)
model = SignedOpinionModel(G, 30, seed=7, comm=metrics["comm"])
ho = SignedOpinionModel(G, 300, seed=999_999, comm=metrics["comm"])
pool = signed_pool(model, metrics["pr"], metrics["core"], 200, seed=11)
st = make_context(G, 0.05, metrics, mode="mr")

for alg in ["Degree", "SignedPageRank"]:
    t0 = time.perf_counter()
    if alg == "Degree":
        S = sorted(range(model.n), key=lambda i: -model.deg[i])[:10]
    else:
        S = list(map(int, sorted(range(model.n),
                                 key=lambda i: -metrics["spr"][i])[:10]))
    pos, neg, net, div, fair = ho.components(S)
    print(f"{alg:16s} pos={pos:8.1f} neg={neg:8.1f} net={net:8.1f} "
          f"div={div:.3f} fair={fair:.3f} t={time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
cfg = ABCConfig(colony_size=8, max_cycles=5, eval_budget=2000, random_state=600)
r = abc_optimize(model, 10, pool, cfg, name="ABC (proposed)")
pos, neg, net, div, fair = ho.components(r.seeds)
print(f"ABC proposed     pos={pos:8.1f} neg={neg:8.1f} net={net:8.1f} "
      f"div={div:.3f} fair={fair:.3f} t={time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
cfg = ABCConfig(colony_size=8, max_cycles=5, eval_budget=2000, random_state=600,
                seed_with_celf=False, gbest_prob=0.0, selection="roulette",
                directed_scout=False, dedupe=False, potential_weight=0.0,
                explore_prob=0.0, accept_equal=False)
r = abc_optimize(model, 10, pool, cfg, name="ABC (plain)")
pos, neg, net, div, fair = ho.components(r.seeds)
print(f"ABC plain        pos={pos:8.1f} neg={neg:8.1f} net={net:8.1f} "
      f"div={div:.3f} fair={fair:.3f} t={time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
S, stats = abcnvs_search(st, 10, seed=0, verbose=False,
                         params=dict(gmax=20, NP=20))
pos, neg, net, div, fair = ho.components(S)
print(f"ABCVNS fixed     pos={pos:8.1f} neg={neg:8.1f} net={net:8.1f} "
      f"div={div:.3f} fair={fair:.3f} t={time.perf_counter()-t0:.1f}s")
print("SMOKE OK")