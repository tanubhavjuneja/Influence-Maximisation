"""
anchor_opinion.py
=================
Measure a small set of cells on the GPU to calibrate the CPU->GPU conversion
of the timings in results_opinion.csv.  Does NOT write to the CSV.

Anchors (same seeds as the original run):
  slashdot       k=10 ABC (proposed) rs=0   -> sigma-workload ratio
  slashdot       k=10 ABC (plain)    rs=0   -> sigma-workload ratio
  bitcoin_alpha  k=10 ABC (proposed) rs=0   -> sigma ratio, small graph
  bitcoin_alpha  k=10 ABCVNS (fixed) rs=0   -> LIE ratio, small graph
  bitcoin_otc    k=10 ABC (proposed) rs=0
  bitcoin_otc    k=10 ABCVNS (fixed) rs=0
"""
import backend as be
from abcvns_paper import make_context
from graphio import load_graph
from metrics import get_metrics, signed_pool, signed_pagerank
from opinion_model import SignedOpinionModel
from run_opinion import P_IC, POOL_SIZE, R_FIT, R_HO, one_cell

CELLS = [
    ("slashdot", 10, "ABC (proposed)", 0),
    ("slashdot", 10, "ABC (plain)", 0),
    ("bitcoin_alpha", 10, "ABC (proposed)", 0),
    ("bitcoin_alpha", 10, "ABCVNS (fixed)", 0),
    ("bitcoin_otc", 10, "ABC (proposed)", 0),
    ("bitcoin_otc", 10, "ABCVNS (fixed)", 0),
]

print(f"backend={'gpu' if be.GPU else 'cpu'}", flush=True)
for graph in ["slashdot", "bitcoin_alpha", "bitcoin_otc"]:
    cells = [c for c in CELLS if c[0] == graph]
    if not cells:
        continue
    G = load_graph(graph, verbose=False)
    m = get_metrics(G)
    m["spr"] = signed_pagerank(G)
    model = SignedOpinionModel(G, R_FIT, seed=7, comm=m["comm"])
    ho = SignedOpinionModel(G, R_HO, seed=999_999, comm=m["comm"])
    pool = signed_pool(model, m["pr"], m["core"], POOL_SIZE, seed=11)
    st = make_context(G, P_IC[graph], m, mode="mr")
    for g, k, alg, rs in cells:
        row = one_cell(g, G, k, alg, rs, model, ho, m, pool, st)
        print(f"ANCHOR {g} k={k} {alg} rs={rs} "
              f"time={row['time_s']:.1f}s pos={row['pos_spread']:.1f}", flush=True)
