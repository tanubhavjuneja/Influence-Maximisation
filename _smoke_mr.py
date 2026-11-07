import time
from real_datasets import load_real_graph
from hybrid_engine import ChunkedMajorityRuleModel, get_metrics
from abcvns import make_abcnvs_context, abcnvs_search

G = load_real_graph("bitcoin_alpha", verbose=False)
metrics = get_metrics("bitcoin_alpha", G)
model = ChunkedMajorityRuleModel(G, "smr", 0.15, 150, seed=7, chunk=256)
ho = ChunkedMajorityRuleModel(G, "smr", 0.15, 1000, seed=999999, chunk=256)

for mode in ["lie", "mr"]:
    st = make_abcnvs_context(G, 0.05, metrics, mode=mode, delta=0.15)
    t0 = time.perf_counter()
    S, stats = abcnvs_search(st, 10, seed=0, verbose=False)
    fit = st.fitness(S)
    print(f"mode={mode:4s} wall={time.perf_counter()-t0:6.1f}s "
          f"fit={fit:9.2f} sigma_in={model.sigma(S):8.1f} "
          f"sigma_ho={ho.sigma(S):8.1f}  seeds={S}")