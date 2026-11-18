"""Find a concrete non-monotonicity witness: a node whose addition to the
optimised k=10 seed set lowers held-out net spread."""
from abc_proposed import ABCConfig, abc_optimize
from graphio import load_graph
from metrics import get_metrics, signed_pool
from opinion_model import SignedOpinionModel

G = load_graph("bitcoin_alpha", verbose=False)
m = get_metrics(G)
model = SignedOpinionModel(G, 30, seed=7, comm=m["comm"])
ho = SignedOpinionModel(G, 300, seed=999_999, comm=m["comm"])
pool = signed_pool(model, m["pr"], m["core"], 200, seed=11)

cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                seed_with_celf=True, eval_budget=10 * 200, random_state=600)
r = abc_optimize(model, 10, pool, cfg, name="ABC")
S = r.seeds
pos0, neg0, net0, div0, fair0 = ho.components(S)
print(f"base: pos={pos0:.1f} neg={neg0:.1f} net={net0:.1f}")

worst = None
for v in pool:
    if v in S:
        continue
    pos, neg, net, div, fair = ho.components(S + [v])
    if worst is None or net < worst[1]:
        worst = (v, net, pos, neg)
if worst and worst[1] < net0:
    v, net, pos, neg = worst
    print(f"witness node {v}: pos={pos:.1f} neg={neg:.1f} net={net:.1f} "
          f"(loss {net0 - net:.1f})")
else:
    print("no witness in this pool")
