r"""
run_experiments.py
==================
Driver for the full empirical study.  Produces everything the report and the
slide deck depend on.

Methodology (v3, review corrections applied)
--------------------------------------------
* EQUAL BUDGET: Greedy's natural cost is exactly k * |pool| fresh evaluations.
  Every budgeted solver (Greedy, Greedy+Phi, CELF+LS, Sandwich-CELF, SA,
  GRASP, ABC, ABC-CELF) is capped at that SAME budget.  CELF alone stays as
  the cheap baseline.
* HELD-OUT SCORING: every final seed set is re-scored on FRESH, independent
  threshold draws with R_ho = 1000 (`spread_heldout`); `spread` is the
  in-sample value the solver optimised.  All comparisons use the held-out
  column.  Activation fractions (held-out spread / n) are reported so that
  relative gains against trivial baselines are not mistaken for substance.
* PAIRED STATISTICS: per-family Wilcoxon signed-rank and bootstrap CIs on
  held-out spreads, over (graph, k) configurations.
* CELF LAZY-VIOLATION RATE: aggregated over ALL CELF runs (weighted by bound
  checks) -- the honest replacement for the earlier "89%" smoke-test figure.
* OPT BIAS: the brute-force optimum is re-scored on the held-out draws, and
  ratios are computed against that held-out OPT, not the in-sample best.

Studies
-------
S1  Structural verification     monotonicity, submodularity, Mossel-Roch test
S2  Exact optimum               brute force on small graphs, held-out ratios
S3  Main comparison             all algorithms x graphs x k, equal budget
S4  Convergence                 best-so-far vs evaluations spent
S5  Delta sweep                 advantage vs non-submodularity (vs CELF AND Greedy)
S8  Signed graphs               same protocol on signed/weighted networks
S9  Budget scaling              0.5x-8x budget sweep

Outputs land in ./results (CSV + JSON) and ./figures (PNG).
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from majority_rule_model import (
    MajorityRuleModel, candidate_pool, load_graph, heldout_model,
    random_seeds, top_degree, top_pagerank, top_kcore, mixed_heuristic,
    greedy_mc, greedy_phi, celf,
)
from abc_celf import ABCConfig, abc_optimize, celf_exchange_refine
from baselines import simulated_annealing, grasp, sandwich_celf
from theory_checks import (
    check_monotonicity, minimal_counterexample, counterexample_survives_stochasticity,
    star_counterexample_sweep, exhaustive_violation_scan, local_function_curvature,
    submodularity_ratio,
)
from signed_rule_model import (
    SignedMajorityRuleModel, signed_candidate_pool,
    signed_hub_blocker, signed_star, signed_cycle, signed_sbm, signed_ba, signed_ws,
)

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
FIGURES = Path("figures"); FIGURES.mkdir(exist_ok=True)

PALETTE = {
    "Random": "#9aa0a6", "Degree": "#7f8c8d", "PageRank": "#95a5a6",
    "k-Core": "#b0b8bd", "MajorityDiscount": "#5d6d7e", "NetSupport": "#5d6d7e",
    "Greedy": "#c0392b", "Greedy+Phi": "#e67e22", "CELF": "#e67e22",
    "CELF + LS": "#f1c40f", "Sandwich-CELF": "#8e44ad",
    "SA": "#2980b9", "GRASP": "#3498db",
    "ABC (no refinement)": "#16a085", "ABC (proposed)": "#27ae60",
    "ABC-CELF (hybrid)": "#2ecc71",
}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})

HELDOUT_SEED = 999999
HELDOUT_R = 1000

FAMILIES = {
    "BA/powerlaw": ("ba500", "ba1000", "ba2000", "powerlaw1000", "powerlaw2000"),
    "WS": ("ws1000", "ws2000"),
    "SBM": ("sbm900",),
    "small": ("karate", "lesmis"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rescore(model, seeds, ho=None) -> float:
    """Held-out score of a seed set on fresh threshold draws (R=1000)."""
    if ho is None:
        ho = type(model)(model.graph, variant=model.variant, delta=model.delta,
                         n_samples=HELDOUT_R, seed=HELDOUT_SEED,
                         backend=model.backend)
    return ho.sigma(seeds)


# --------------------------------------------------------------------------- #
# S1 - structural verification
# --------------------------------------------------------------------------- #
def study_structure(only: str | None = None):
    log("S1  structural verification")
    out = {}

    out["minimal_counterexample"] = minimal_counterexample(verbose=False)
    out["delta_sweep_counterexample"] = counterexample_survives_stochasticity()
    out["delta_sweep_counterexample_K19"] = star_counterexample_sweep(
        9, 4, deltas=(0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.49, 0.5))
    out["local_function"] = [
        local_function_curvature(v, d, 10)
        for v, d in (("lt", 0.5), ("smr", 0.30), ("smr", 0.15), ("dmr", 0.0))
    ]

    rows = []
    specs = [("karate", 4000, 16, 600, 250), ("lesmis", 3000, 15, 600, 250),
             ("ba500", 800, 11, 250, 150)]
    if only:
        specs = [x for x in specs if x[0] == only]
    for gname, R, cap, mono_n, ratio_n in specs:
        G = load_graph(gname)
        for variant, delta in (("lt", 0.5), ("smr", 0.30), ("smr", 0.15), ("dmr", 0.0)):
            m = MajorityRuleModel(G, variant, delta, n_samples=R, seed=7)
            mono = check_monotonicity(m, trials=mono_n, seed=3)
            sub = exhaustive_violation_scan(m, max_T=3, node_cap=cap, seed=3)
            pool = candidate_pool(m, min(m.n, 100))
            gam = submodularity_ratio(m, k=5, pool=pool, samples=ratio_n, seed=3).as_dict()
            rows.append({
                "graph": gname, "n": m.n, "m": G.number_of_edges(),
                "variant": variant, "delta": delta, "R": R,
                "monotone": mono["holds"], "mono_violations": mono["violations"],
                "submod_triples": sub["triples_tested"],
                "submod_violations": sub["violations"],
                "submod_violation_rate": sub["violation_rate"],
                "nondeg_violation_rate": sub["nondegenerate_violation_rate"],
                "tolerance": sub["tolerance"],
                "worst_violation": sub["worst_violation"],
                **gam,
            })
            log(f"    {gname:8s} {variant}(d={delta:.2f})  "
                f"monotone={mono['holds']}  "
                f"submod_viol={sub['violation_rate']:.1%} "
                f"(non-degenerate {sub['nondegenerate_violation_rate']:.1%})  "
                f"gamma_min={gam['gamma_min']:.3f}")
    df = pd.DataFrame(rows)
    tag = f"_{only}" if only else ""
    df.to_csv(RESULTS / f"s1_structure{tag}.csv", index=False)
    out["table"] = rows
    (RESULTS / f"s1_structure{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    return df, out


# --------------------------------------------------------------------------- #
# S2 - exact optimum by brute force (held-out corrected)
# --------------------------------------------------------------------------- #
def study_exact(graphs=(("karate", 4), ("lesmis", 3), ("florentine", 4),
                       ("davis", 3), ("star60", 3), ("cycle60", 3))):
    log("S2  exact optimum (brute force, held-out corrected)")
    rows = []
    for gname, k in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, "smr", 0.15, n_samples=200, seed=7)
        nodes = list(range(m.n))
        best, best_set = -1.0, None
        t0 = time.perf_counter()
        for S in itertools.combinations(nodes, k):
            v = float(m._simulate(S).mean())
            if v > best:
                best, best_set = v, S
        bt = time.perf_counter() - t0
        opt_ho = rescore(m, best_set)
        log(f"    {gname} k={k}: OPT={best:.3f} (held-out {opt_ho:.3f}) in {bt:.1f}s")

        m.reset_counters(); m.clear_cache()
        pool = candidate_pool(m, min(m.n, 80))
        budget = k * len(pool)
        algos = {}
        algos["Degree"] = top_degree(m, k)
        algos["PageRank"] = top_pagerank(m, k)
        algos["MajorityDiscount"] = mixed_heuristic(m, k)
        algos["Greedy"] = greedy_mc(m, k, pool=pool)
        algos["CELF"] = celf(m, k, pool=pool)
        S2, f2, _ = celf_exchange_refine(m, algos["CELF"].seeds, pool, max_passes=4,
                                         eval_budget=max(0, budget - algos["CELF"].evaluations))
        cfg = ABCConfig(colony_size=24, max_cycles=40, refine_every=0,
                        seed_with_celf=True, eval_budget=budget, random_state=1)
        r_abc = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")

        for name, seeds, sp in [
            ("Degree", algos["Degree"].seeds, algos["Degree"].spread),
            ("PageRank", algos["PageRank"].seeds, algos["PageRank"].spread),
            ("MajorityDiscount", algos["MajorityDiscount"].seeds, algos["MajorityDiscount"].spread),
            ("Greedy", algos["Greedy"].seeds, algos["Greedy"].spread),
            ("CELF", algos["CELF"].seeds, algos["CELF"].spread),
            ("CELF + LS", S2, f2),
            ("ABC (proposed)", r_abc.seeds, r_abc.spread),
        ]:
            ho = rescore(m, seeds)
            rows.append({"graph": gname, "k": k, "algorithm": name,
                         "spread": sp, "spread_heldout": ho,
                         "OPT": best, "OPT_heldout": opt_ho,
                         "ratio": sp / best, "ratio_heldout": ho / opt_ho})
            log(f"      {name:22s} sigma={sp:8.3f}  ratio={sp/best:.4f}  "
                f"ratio_ho={ho/opt_ho:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "s2_exact.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# S3 - main comparison, equal budget + held-out + set-level baselines
# --------------------------------------------------------------------------- #
def study_main(graphs=("karate", "lesmis", "ba500", "ws1000", "powerlaw1000",
                       "sbm900", "ba1000", "ba2000", "ws2000", "powerlaw2000"),
               ks=(5, 10, 20, 30),
               variant="smr", delta=0.15, R=150, restarts=3):
    log("S3  main comparison (equal budget + held-out R=1000 + set-level baselines)")
    rows, curves = [], []
    for gname in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, variant, delta, n_samples=R, seed=7)
        pool = candidate_pool(m, min(m.n, 200))
        ho = type(m)(m.graph, variant=variant, delta=delta,
                     n_samples=HELDOUT_R, seed=HELDOUT_SEED, backend=m.backend)
        log(f"  {m.describe()}  |pool|={len(pool)}")
        for k in ks:
            if k >= m.n // 2:
                continue
            budget = k * len(pool)          # greedy's exact cost = the shared budget

            # ---- deterministic baselines ---- #
            for fn in (random_seeds, top_degree, top_pagerank, top_kcore, mixed_heuristic):
                m.reset_counters(); m.clear_cache()
                r = fn(m, k)
                rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                             **r.as_row(), "spread_heldout": rescore(m, r.seeds, ho)})

            # ---- budgeted baselines, each <= budget ---- #
            m.reset_counters(); m.clear_cache()
            r_greedy = greedy_mc(m, k, pool=pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_greedy.as_row(), "spread_heldout": rescore(m, r_greedy.seeds, ho)})

            m.reset_counters(); m.clear_cache()
            r_gphi = greedy_phi(m, k, pool=pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_gphi.as_row(), "spread_heldout": rescore(m, r_gphi.seeds, ho)})

            m.reset_counters(); m.clear_cache()
            r_celf = celf(m, k, pool=pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_celf.as_row(), "spread_heldout": rescore(m, r_celf.seeds, ho)})

            m.reset_counters(); m.clear_cache()
            t0 = time.perf_counter()
            S2, f2, st = celf_exchange_refine(
                m, r_celf.seeds, pool, max_passes=4,
                eval_budget=max(0, budget - r_celf.evaluations))
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         "algorithm": "CELF + LS", "spread": f2,
                         "spread_heldout": rescore(m, S2, ho),
                         "runtime_s": time.perf_counter() - t0,
                         "evaluations": m.n_evals})

            m.reset_counters(); m.clear_cache()
            r_sand = sandwich_celf(m, k, pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_sand.as_row(), "spread_heldout": rescore(m, r_sand.seeds, ho)})

            m.reset_counters(); m.clear_cache()
            r_sa = simulated_annealing(m, k, pool, eval_budget=budget,
                                       restart=True, seed=1)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_sa.as_row(), "spread_heldout": rescore(m, r_sa.seeds, ho)})

            m.reset_counters(); m.clear_cache()
            r_gr = grasp(m, k, pool, eval_budget=budget, seed=1)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_gr.as_row(), "spread_heldout": rescore(m, r_gr.seeds, ho)})

            # ---- stochastic solvers, multiple restarts ---- #
            for rs in range(restarts):
                m.reset_counters(); m.clear_cache()
                cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                                seed_with_celf=False, eval_budget=budget,
                                random_state=100 + rs)
                r = abc_optimize(m, k, pool, cfg, name="ABC (no refinement)")
                rows.append({"graph": gname, "n": m.n, "k": k, "restart": rs,
                             **r.as_row(), "spread_heldout": rescore(m, r.seeds, ho)})

                m.reset_counters(); m.clear_cache()
                cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                                seed_with_celf=True, eval_budget=budget,
                                random_state=100 + rs)
                rp = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                rows.append({"graph": gname, "n": m.n, "k": k, "restart": rs,
                             **rp.as_row(), "spread_heldout": rescore(m, rp.seeds, ho)})

                m.reset_counters(); m.clear_cache()
                cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=10,
                                seed_with_celf=True, eval_budget=budget,
                                random_state=100 + rs)
                r2 = abc_optimize(m, k, pool, cfg, name="ABC-CELF (hybrid)")
                rows.append({"graph": gname, "n": m.n, "k": k, "restart": rs,
                             **r2.as_row(), "spread_heldout": rescore(m, r2.seeds, ho)})
                if rs == 0 and k in (10, 20):
                    curves.append({"graph": gname, "k": k,
                                   "algorithm": "ABC-CELF (hybrid)",
                                   "evals": r2.extra["eval_curve"], "best": r2.trace})
                    curves.append({"graph": gname, "k": k,
                                   "algorithm": "ABC (no refinement)",
                                   "evals": r.extra["eval_curve"], "best": r.trace})
            log(f"    k={k:3d}  Greedy={r_greedy.spread:7.2f}  CELF={r_celf.spread:7.2f}  "
                f"SA={r_sa.spread:7.2f}  ABC-CELF={r2.spread:7.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "s3_main.csv", index=False)
    (RESULTS / "s4_curves.json").write_text(json.dumps(curves))
    return df, curves


# --------------------------------------------------------------------------- #
# S5 - delta sweep (vs CELF and Greedy, Spearman with structure)
# --------------------------------------------------------------------------- #
def study_delta(gname="ba500", k=15, deltas=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
                R=200, restarts=3):
    log(f"S5  delta sweep on {gname} (submodularity <-> advantage)")
    G = load_graph(gname)
    rows = []
    for d in deltas:
        variant = "dmr" if d == 0.0 else ("lt" if d >= 0.5 else "smr")
        m_struct = MajorityRuleModel(G, variant, d, n_samples=1500, seed=7)
        sub = exhaustive_violation_scan(m_struct, max_T=3, node_cap=12, seed=3)
        gam = submodularity_ratio(m_struct, k=5,
                                  pool=candidate_pool(m_struct, 80),
                                  samples=200, seed=3).as_dict()
        del m_struct

        m = MajorityRuleModel(G, variant, d, n_samples=R, seed=7)
        pool = candidate_pool(m, 150)
        budget = k * len(pool)

        m.reset_counters(); m.clear_cache()
        r_celf = celf(m, k, pool=pool)
        m.reset_counters(); m.clear_cache()
        r_greedy = greedy_mc(m, k, pool=pool)
        celf_ho, greedy_ho = rescore(m, r_celf.seeds), rescore(m, r_greedy.seeds)
        abc_runs, abc_ho = [], []
        for rs in range(restarts):
            m.reset_counters(); m.clear_cache()
            cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                            seed_with_celf=True, eval_budget=budget, random_state=200 + rs)
            r = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
            abc_runs.append(r.spread)
            abc_ho.append(rescore(m, r.seeds))
        rows.append({
            "delta": d, "variant": variant,
            "submod_violation_rate": sub["violation_rate"],
            "nondeg_violation_rate": sub["nondegenerate_violation_rate"],
            "gamma_min": gam["gamma_min"], "gamma_mean": gam["gamma_mean"],
            "celf": r_celf.spread, "celf_heldout": celf_ho,
            "greedy": r_greedy.spread, "greedy_heldout": greedy_ho,
            "abc_celf_mean": float(np.mean(abc_runs)),
            "abc_celf_best": max(abc_runs),
            "abc_celf_heldout_mean": float(np.mean(abc_ho)),
            "improvement_pct": 100 * (float(np.mean(abc_ho)) - celf_ho) / max(celf_ho, 1e-9),
            "improvement_vs_greedy_pct": 100 * (float(np.mean(abc_ho)) - greedy_ho) / max(greedy_ho, 1e-9),
            "celf_lazy_violation_rate": r_celf.extra["lazy_violation_rate"],
        })
        log(f"    delta={d:.2f} [{variant}]  viol={sub['violation_rate']:.1%}  "
            f"gamma={gam['gamma_min']:.3f}  CELF={celf_ho:.2f}  "
            f"Greedy={greedy_ho:.2f}  ABC-CELF={np.mean(abc_ho):.2f}  "
            f"(+{rows[-1]['improvement_vs_greedy_pct']:.1f}% vs Greedy)")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"s5_delta_{gname}.csv", index=False)
    rho_g = sps.spearmanr(df.delta, df.improvement_vs_greedy_pct)
    rho_c = sps.spearmanr(df.delta, df.improvement_pct)
    rho_v = sps.spearmanr(df.delta, df.gamma_min)
    log(f"  Spearman: adv-vs-Greedy ~ delta: rho={rho_g.statistic:.3f} (p={rho_g.pvalue:.3g})")
    log(f"  Spearman: adv-vs-CELF  ~ delta: rho={rho_c.statistic:.3f} (p={rho_c.pvalue:.3g})")
    log(f"  Spearman: gamma_min    ~ delta: rho={rho_v.statistic:.3f} (p={rho_v.pvalue:.3g})")
    # the claim is about SUBMODULARITY, so correlate the gain with the
    # structural measures directly (delta is only a proxy)
    rho_gg = sps.spearmanr(df.gamma_min, df.improvement_vs_greedy_pct)
    rho_gv = sps.spearmanr(df.nondeg_violation_rate, df.improvement_vs_greedy_pct)
    rho_cg = sps.spearmanr(df.gamma_min, df.improvement_pct)
    log(f"  Spearman: adv-vs-Greedy ~ gamma_min: rho={rho_gg.statistic:.3f} (p={rho_gg.pvalue:.3g})")
    log(f"  Spearman: adv-vs-Greedy ~ viol_rate: rho={rho_gv.statistic:.3f} (p={rho_gv.pvalue:.3g})")
    log(f"  Spearman: adv-vs-CELF   ~ gamma_min: rho={rho_cg.statistic:.3f} (p={rho_cg.pvalue:.3g})")
    return df


# --------------------------------------------------------------------------- #
# S8 - signed graphs
# --------------------------------------------------------------------------- #
def _signed_graphs():
    return {
        "signed_hub": signed_hub_blocker(),
        "signed_star7": signed_star(5, 2),
        "signed_cycle40": signed_cycle(40, 0.25),
        "signed_sbm120": signed_sbm((60, 60), p_in=0.10, p_out=0.02, seed=11),
        "signed_sbm450": signed_sbm((150, 150, 150), p_in=0.08, p_out=0.015,
                                    p_neg_in=0.08, p_neg_out=0.7, seed=11),
        "signed_ws300": signed_ws(300, 8, 0.1, p_neg=0.3, seed=12),
        "signed_ba500": signed_ba(500, 4, p_neg=0.35, seed=13),
    }


def signed_net_support(model, k) -> "SelectionResult":
    from majority_rule_model import SelectionResult
    import time as _t
    t0, e0 = _t.perf_counter(), model.n_evals
    s = sorted(range(model.n), key=lambda i: -model.pool_weight[i])[:k]
    return SelectionResult(algorithm="NetSupport", seeds=s,
                           spread=model.sigma(s), runtime=_t.perf_counter() - t0,
                           evaluations=model.n_evals - e0)


def study_signed(variant="smr", delta=0.15, R=200, restarts=3, ks=(5, 10, 15)):
    log("S8  signed graphs (equal budget + held-out)")
    rows = []
    for gname, G in _signed_graphs().items():
        m = SignedMajorityRuleModel(G, variant=variant, delta=delta,
                                    n_samples=R, seed=7)
        mono = check_monotonicity(m, trials=400, seed=3)
        log(f"  {m.describe()}  monotone={mono['holds']} "
            f"({mono['violations']}/{mono['trials']})")
        pool = signed_candidate_pool(m, min(m.n, 150))
        for k in ks:
            if k >= m.n // 2:
                continue
            budget = k * len(pool)

            m.reset_counters(); m.clear_cache()
            r = signed_net_support(m, k)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r.as_row(), "spread_heldout": rescore(m, r.seeds)})

            m.reset_counters(); m.clear_cache()
            r_greedy = greedy_mc(m, k, pool=pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_greedy.as_row(), "spread_heldout": rescore(m, r_greedy.seeds)})

            m.reset_counters(); m.clear_cache()
            r_celf = celf(m, k, pool=pool)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_celf.as_row(), "spread_heldout": rescore(m, r_celf.seeds)})

            m.reset_counters(); m.clear_cache()
            t0 = time.perf_counter()
            S2, f2, _ = celf_exchange_refine(
                m, r_celf.seeds, pool, max_passes=4,
                eval_budget=max(0, budget - r_celf.evaluations))
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         "algorithm": "CELF + LS", "spread": f2,
                         "spread_heldout": rescore(m, S2),
                         "runtime_s": time.perf_counter() - t0,
                         "evaluations": m.n_evals})

            m.reset_counters(); m.clear_cache()
            r_sa = simulated_annealing(m, k, pool, eval_budget=budget,
                                       restart=True, seed=1)
            rows.append({"graph": gname, "n": m.n, "k": k, "restart": 0,
                         **r_sa.as_row(), "spread_heldout": rescore(m, r_sa.seeds)})

            for rs in range(restarts):
                m.reset_counters(); m.clear_cache()
                cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                                seed_with_celf=True, eval_budget=budget,
                                random_state=300 + rs)
                r2 = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                rows.append({"graph": gname, "n": m.n, "k": k, "restart": rs,
                             **r2.as_row(), "spread_heldout": rescore(m, r2.seeds)})
            log(f"    {gname:16s} k={k}  Greedy={r_greedy.spread:7.2f}  "
                f"CELF={r_celf.spread:7.2f}  SA={r_sa.spread:7.2f}  "
                f"ABC-CELF={r2.spread:7.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "s8_signed.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# S9 - budget scaling
# --------------------------------------------------------------------------- #
def study_budget(graphs=("ba500", "ws1000"), k=15, mults=(0.5, 1, 4, 8),
                 R=150, restarts=1):
    log("S9  budget scaling")
    out_path = RESULTS / "s9_budget.csv"
    for gname in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, "smr", 0.15, n_samples=R, seed=7)
        pool = candidate_pool(m, 200)
        base = k * len(pool)
        for mult in mults:
            budget = int(base * mult)
            m.reset_counters(); m.clear_cache()
            r_g = greedy_mc(m, k, pool=pool)
            gh = rescore(m, r_g.seeds)
            m.reset_counters(); m.clear_cache()
            r_c = celf(m, k, pool=pool)
            ch = rescore(m, r_c.seeds)
            ah, sah = [], []
            for rs in range(restarts):
                m.reset_counters(); m.clear_cache()
                # max_cycles must scale with budget, else ABC flatlines
                cfg = ABCConfig(colony_size=30, max_cycles=max(150, budget // 15),
                                refine_every=0, seed_with_celf=True,
                                eval_budget=budget, random_state=400 + rs)
                r = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                ah.append(rescore(m, r.seeds))
                m.reset_counters(); m.clear_cache()
                r_sa = simulated_annealing(m, k, pool, eval_budget=budget,
                                           restart=True, seed=rs + 1)
                sah.append(rescore(m, r_sa.seeds))
            row = {"graph": gname, "mult": mult, "budget": budget,
                   "greedy_ho": gh, "celf_ho": ch,
                   "abc_ho_mean": float(np.mean(ah)),
                   "sa_ho_mean": float(np.mean(sah))}
            df = pd.DataFrame([row])
            df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            log(f"  {gname} budget={budget:6d}  Greedy={gh:7.2f}  CELF={ch:7.2f}  "
                f"ABC={np.mean(ah):7.2f}  SA={np.mean(sah):7.2f}   [saved]")
    return pd.read_csv(out_path)


# --------------------------------------------------------------------------- #
# S10 - multi-instance family study (graph = statistical unit)
# --------------------------------------------------------------------------- #
def _multi_instances():
    """8 fresh instances per family, varying generator seeds AND structural
    parameters (degree / attachment / density), so the family-level claims do
    not rest on one or two hand-picked graphs."""
    import networkx as nx
    out = {}
    ws = []
    for i in range(8):
        deg = 6 + 2 * (i % 3)
        G = nx.watts_strogatz_graph(1000, deg, 0.1, seed=300 + i)
        ws.append((f"ws_{deg}_{i}", G))
    ba = []
    for i in range(8):
        m0 = 3 + (i % 3)
        G = nx.barabasi_albert_graph(1000, m0, seed=400 + i)
        ba.append((f"ba_{m0}_{i}", G))
    sbm = []
    for i in range(8):
        p_in = 0.03 + 0.01 * (i % 3)
        sizes = [300, 300, 300]
        p = np.full((3, 3), p_in * 0.1)
        np.fill_diagonal(p, p_in)
        G = nx.stochastic_block_model(sizes, p, seed=500 + i)
        sbm.append((f"sbm_{p_in}_{i}", G))
    out["WS"] = ws
    out["BA"] = ba
    out["SBM"] = sbm
    return out


MULTI_SOLVERS = ("Greedy", "CELF", "CELF + LS", "SA", "ABC (proposed)",
                 "BFSBall", "DenseCluster")


def study_multi(families=("WS", "BA", "SBM"), ks=(5, 10, 20), R=100,
                restarts=1):
    """
    S10: replicate the family comparison on 8 fresh instances per family.
    Every solver gets the shared budget k * |pool|.  The GRAPH is the unit of
    analysis: spreads are averaged over k within a graph before any paired
    test, so correlated k-configs of the same graph cannot masquerade as
    independent evidence.
    """
    from majority_rule_model import bfs_ball, dense_cluster
    log("S10  multi-instance family study (8 instances per family, graph = unit)")
    out_path = RESULTS / "s10_multi.csv"
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.graph, old.k, old.algorithm))
    for fam in families:
        for gname, G in _multi_instances()[fam]:
            m = MajorityRuleModel(G, "smr", 0.15, n_samples=R, seed=7)
            ho = type(m)(G, variant="smr", delta=0.15, n_samples=HELDOUT_R,
                         seed=HELDOUT_SEED, backend=m.backend)
            pool = candidate_pool(m, min(m.n, 200))
            for k in ks:
                budget = k * len(pool)
                for alg in MULTI_SOLVERS:
                    if (gname, k, alg) in done:
                        continue
                    t0 = time.perf_counter()
                    m.reset_counters(); m.clear_cache()
                    if alg == "Greedy":
                        r = greedy_mc(m, k, pool=pool)
                    elif alg == "CELF":
                        r = celf(m, k, pool=pool)
                    elif alg == "CELF + LS":
                        rc = celf(m, k, pool=pool)
                        S2, f2, _ = celf_exchange_refine(
                            m, rc.seeds, pool, max_passes=4,
                            eval_budget=max(0, budget - rc.evaluations))
                        r = type(rc)(algorithm="CELF + LS", seeds=S2, spread=f2,
                                     runtime=0.0, evaluations=m.n_evals)
                    elif alg == "SA":
                        r = simulated_annealing(m, k, pool, eval_budget=budget,
                                                restart=True, seed=1)
                    elif alg == "ABC (proposed)":
                        best = None
                        for rs in range(restarts):
                            m.reset_counters(); m.clear_cache()
                            cfg = ABCConfig(colony_size=30, max_cycles=150,
                                            refine_every=0, seed_with_celf=True,
                                            eval_budget=budget,
                                            random_state=600 + rs)
                            rr = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                            if best is None or rr.spread > best.spread:
                                best = rr
                        r = best
                    elif alg == "BFSBall":
                        r = bfs_ball(m, k)
                    elif alg == "DenseCluster":
                        r = dense_cluster(m, k)
                    wall = time.perf_counter() - t0
                    hov = rescore(m, r.seeds, ho)
                    row = {"family": fam, "graph": gname, "k": k,
                           "algorithm": alg, "budget": budget,
                           "spread": float(r.spread), "heldout": float(hov),
                           "wall_s": float(wall),
                           "evaluations": int(r.evaluations)}
                    pd.DataFrame([row]).to_csv(out_path, mode="a",
                                               header=not out_path.exists(), index=False)
                    log(f"    {gname:12s} k={k:2d} {alg:16s} ho={hov:7.2f} "
                        f"({wall:.0f}s)  [saved]")
    df = pd.read_csv(out_path)

    # ---- graph-level aggregation: the unit of analysis ---- #
    glev = df.groupby(["family", "graph", "algorithm"]).agg(
        heldout=("heldout", "mean"), wall=("wall_s", "sum")).reset_index()
    piv = glev.pivot_table(index=["family", "graph"], columns="algorithm",
                           values="heldout")
    piv.to_csv(RESULTS / "s10_multi_graphlevel.csv")

    log("\nS10 graph-level family means (each graph averaged over k):")
    fams = piv.groupby(level="family").mean()
    for fam in families:
        cols = [c for c in MULTI_SOLVERS if c in fams.columns]
        log(f"  {fam:4s} " + "  ".join(f"{c}={fams.loc[fam, c]:.2f}" for c in cols))

    stats = {}
    for fam in families:
        sub = piv.loc[fam] if fam in piv.index.get_level_values("family") else None
        if sub is None or sub.empty:
            continue
        abc = sub["ABC (proposed)"]
        stats[f"{fam}_abc_vs_greedy"] = _graph_paired(abc, sub["Greedy"])
        stats[f"{fam}_abc_vs_sa"] = _graph_paired(abc, sub["SA"])
        stats[f"{fam}_abc_vs_celf"] = _graph_paired(abc, sub["CELF"])
        stats[f"{fam}_sa_vs_bfs"] = _graph_paired(sub["SA"], sub.get("BFSBall"))
        stats[f"{fam}_means"] = {c: round(float(sub[c].mean()), 3) for c in sub.columns}
    (RESULTS / "s10_multi_stats.json").write_text(
        json.dumps(stats, indent=2, default=str))
    log(json.dumps(stats, indent=2, default=str))
    return df, stats


def _graph_paired(a, b):
    """Paired test with the GRAPH as unit: Wilcoxon + cluster bootstrap of
    the mean difference (resampling graphs, not configs)."""
    a, b = a.dropna(), b.dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    if len(a) < 2:
        return {"n_graphs": len(a), "mean_diff": None, "median_diff": None,
                "wilcoxon_p": None, "ci": None, "wins": 0}
    d = (a - b).values
    w = sps.wilcoxon(d, zero_method="zsplit")
    rng = np.random.default_rng(0)
    boot = np.array([np.mean(rng.choice(d, size=len(d), replace=True))
                     for _ in range(4000)])
    return {"n_graphs": len(a),
            "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)),
            "wilcoxon_p": float(w.pvalue),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((d > 0).sum())}


# --------------------------------------------------------------------------- #
# S11 - sign-ablation control
# --------------------------------------------------------------------------- #
def study_sign_control(R=100, ks=(10, 15), restarts=2):
    """
    S11: same topology with all-positive edges vs signed edges.  If ABC's
    advantage over Greedy persists on the all-positive twin, the effect is the
    plateau, not the signs.
    """
    import networkx as nx
    log("S11  sign-ablation control (signed vs all-positive twin)")
    rows = []
    out_path = RESULTS / "s11_signcontrol.csv"
    for gname, G in _signed_graphs().items():
        if gname not in ("signed_sbm120", "signed_sbm450", "signed_ws300"):
            continue
        Gpos = nx.Graph()
        Gpos.add_nodes_from(G.nodes())
        for u, v in G.edges():
            Gpos.add_edge(u, v)
        for label, GG, Model in (("signed", G, SignedMajorityRuleModel),
                                 ("positive", Gpos, MajorityRuleModel)):
            m = Model(GG, variant="smr", delta=0.15, n_samples=R, seed=7)
            ho = type(m)(GG, variant="smr", delta=0.15, n_samples=HELDOUT_R,
                         seed=HELDOUT_SEED, backend=m.backend)
            pool = (signed_candidate_pool if label == "signed"
                    else candidate_pool)(m, min(m.n, 150))
            for k in ks:
                budget = k * len(pool)
                m.reset_counters(); m.clear_cache()
                rg = greedy_mc(m, k, pool=pool)
                gh = rescore(m, rg.seeds, ho)
                sah, ah = [], []
                for rs in range(restarts):
                    m.reset_counters(); m.clear_cache()
                    r_sa = simulated_annealing(m, k, pool, eval_budget=budget,
                                               restart=True, seed=rs + 1)
                    sah.append(rescore(m, r_sa.seeds, ho))
                    m.reset_counters(); m.clear_cache()
                    cfg = ABCConfig(colony_size=30, max_cycles=150, refine_every=0,
                                    seed_with_celf=True, eval_budget=budget,
                                    random_state=700 + rs)
                    r_ab = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                    ah.append(rescore(m, r_ab.seeds, ho))
                rows.append({"graph": gname, "k": k, "edges": label,
                             "greedy_ho": gh, "sa_ho": float(np.mean(sah)),
                             "abc_ho": float(np.mean(ah)),
                             "abc_vs_greedy_pct": 100 * (np.mean(ah) - gh) / max(gh, 1e-9)})
        log(f"  {gname}: signed vs positive-twin rows appended")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(df.round(2).to_string())
    return df


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _spread_col(df):
    return "spread_heldout" if "spread_heldout" in df.columns else "spread"


def fig_spread_vs_k(df):
    col = _spread_col(df)
    graphs = [g for g in df["graph"].unique()]
    ncol = 3
    nrow = int(np.ceil(len(graphs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.5 * nrow), squeeze=False)
    order = ["Degree", "Greedy", "Greedy+Phi", "CELF", "CELF + LS",
             "SA", "ABC (no refinement)", "ABC (proposed)", "ABC-CELF (hybrid)"]
    for ax, g in zip(axes.ravel(), graphs):
        sub = df[df.graph == g]
        for alg in order:
            s = sub[sub.algorithm == alg].groupby("k")[col].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, marker="o", ms=4, lw=1.9,
                    label=alg, color=PALETTE.get(alg),
                    ls="--" if alg in ("Degree", "Greedy") else "-")
        ax.set_title(g, fontweight="bold")
        ax.set_xlabel("seed set size $k$"); ax.set_ylabel(r"held-out spread $\sigma(S)$")
    for ax in axes.ravel()[len(graphs):]:
        ax.axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Held-out influence spread under the stochastic majority rule "
                 r"($\theta\sim U[0.35,0.65]$)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(FIGURES / "fig1_spread_vs_k.png", bbox_inches="tight")
    plt.close(fig)


def fig_convergence(curves):
    sel = [c for c in curves if c["k"] == 20] or [c for c in curves if c["k"] == 10]
    graphs = sorted({c["graph"] for c in sel})[:3]
    fig, axes = plt.subplots(1, len(graphs), figsize=(4.6 * len(graphs), 3.6), squeeze=False)
    for ax, g in zip(axes[0], graphs):
        for c in sel:
            if c["graph"] != g:
                continue
            n_pts = len(c["evals"])
            ax.plot(c["evals"], c["best"], lw=2, marker="o",
                    ms=6 if n_pts < 20 else 2.2,
                    color=PALETTE.get(c["algorithm"]), label=c["algorithm"])
        ax.set_title(f"{g}  (k={sel[0]['k']})", fontweight="bold")
        ax.set_xlabel(r"$\sigma(\cdot)$ evaluations spent")
        ax.set_ylabel("best spread found")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Anytime convergence: markers = actual checkpoints.", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURES / "fig3_convergence.png", bbox_inches="tight")
    plt.close(fig)


def fig_delta(df, gname):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(df.delta, df.improvement_vs_greedy_pct, "o-", lw=2.4, color="#16a085",
            label="ABC-CELF held-out gain over Greedy (%)")
    ax.plot(df.delta, df.improvement_pct, "s--", lw=2, color="#e67e22",
            label="ABC-CELF held-out gain over CELF (%)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"threshold dispersion $\delta$   "
                  r"($\theta\sim U[\frac{1}{2}-\delta,\frac{1}{2}+\delta]$)")
    ax.set_ylabel("spread gain (%)", color="#16a085")
    ax.tick_params(axis="y", colors="#16a085")

    ax2 = ax.twinx()
    ax2.plot(df.delta, df.gamma_min, "^:", lw=2, color="#c0392b",
             label=r"submodularity ratio $\gamma_{\min}$")
    ax2.plot(df.delta, df.nondeg_violation_rate, "v:", lw=2, color="#7f8c8d",
             label="fraction of triples violating submodularity")
    ax2.set_ylabel(r"$\gamma_{\min}$  /  violation rate")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    ax.set_title(f"Advantage of the metaheuristic vs loss of submodularity ({gname})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_delta_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig_exact(df):
    fig, axes = plt.subplots(1, df.graph.nunique(),
                             figsize=(5.0 * df.graph.nunique(), 4.0), squeeze=False)
    for ax, g in zip(axes[0], df.graph.unique()):
        sub = df[df.graph == g].sort_values("ratio_heldout")
        colors = [PALETTE.get(a, "#888") for a in sub.algorithm]
        ax.barh(sub.algorithm, sub.ratio_heldout, color=colors)
        ax.axvline(1.0, color="k", lw=1.2)
        ax.axvline(1 - np.exp(-1), color="#c0392b", ls="--", lw=1.4)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel(r"$\sigma_{ho}(S)\,/\,\sigma_{ho}(S^{*})$")
        ax.set_title(f"{g}  (k={sub.k.iloc[0]}, held-out OPT)", fontweight="bold")
    fig.suptitle("Approximation ratio against the true optimum (held-out)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_exact_ratio.png", bbox_inches="tight")
    plt.close(fig)


def fig_structure(s1):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for v, d in (("lt", 0.5), ("smr", 0.30), ("smr", 0.15), ("dmr", 0.0)):
        r = local_function_curvature(v, d, 20)
        x = np.arange(len(r["F"])) / (len(r["F"]) - 1)
        lbl = "LT  " + r"$\theta\sim U[0,1]$" if v == "lt" else (
            "majority (deterministic)" if v == "dmr" else rf"majority $\delta={d}$")
        ax.plot(x, r["F"], lw=2.2, marker="o", ms=3, label=lbl)
    ax.set_xlabel("fraction of neighbours active")
    ax.set_ylabel(r"$f_v$ = activation probability")
    ax.set_title("Local activation function", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.03, 0.93, "concave  $\\Rightarrow$  submodular\n(Mossel & Roch 2007, sufficient)",
            transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[1]
    sub = s1[s1.graph == "ba500"] if "ba500" in set(s1.graph) else s1
    lbl = [f"{v}\n$\\delta$={d}" for v, d in zip(sub.variant, sub.delta)]
    ax.bar(lbl, sub.nondeg_violation_rate, color="#c0392b", alpha=0.85,
           label="submodularity violation rate")
    ax.plot(lbl, sub.gamma_min, "o-", color="#16a085", lw=2,
            label=r"submodularity ratio $\gamma_{\min}$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("rate  /  ratio")
    ax.set_title("Structural degradation with a majority threshold", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_structure.png", bbox_inches="tight")
    plt.close(fig)


def fig_budget(df):
    fig, axes = plt.subplots(1, df.graph.nunique(), figsize=(5.5 * df.graph.nunique(), 4), squeeze=False)
    for ax, g in zip(axes[0], df.graph.unique()):
        sub = df[df.graph == g]
        ax.plot(sub.budget, sub.abc_ho_mean, "o-", lw=2, color="#16a085", label="ABC")
        ax.plot(sub.budget, sub.sa_ho_mean, "s-", lw=2, color="#2980b9", label="SA")
        ax.plot(sub.budget, sub.greedy_ho, "^-", lw=2, color="#c0392b", label="Greedy")
        ax.plot(sub.budget, sub.celf_ho, "v-", lw=2, color="#e67e22", label="CELF")
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("held-out spread")
        ax.set_title(g, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out spread vs evaluation budget (x = k * |pool|)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_budget.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Summary statistics
# --------------------------------------------------------------------------- #
def _family_of(g):
    for fam, gs in FAMILIES.items():
        if g in gs:
            return fam
    return "other"


def paired_stats(df, alg, against):
    """Wilcoxon + bootstrap CI of held-out mean gain, over (graph,k) configs."""
    col = _spread_col(df)
    a = df[df.algorithm == alg].groupby(["graph", "k"])[col].mean()
    b = df[df.algorithm == against].groupby(["graph", "k"])[col].mean()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    if not len(a):
        return {"n_configs": 0, "mean_abs_gain": None, "median_abs_gain": None,
                "mean_rel_gain_pct": None, "median_rel_gain_pct": None,
                "max_rel_gain_pct": None, "wins": 0, "ties": 0, "losses": 0,
                "wilcoxon_p": None, "bootstrap_ci": None}
    diff = (a - b).values
    rel = 100 * diff / b.values
    w = sps.wilcoxon(diff, zero_method="zsplit") if len(diff) else None
    rng = np.random.default_rng(0)
    boot = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True))
                     for _ in range(4000)])
    return {
        "n_configs": len(diff),
        "mean_abs_gain": float(np.mean(diff)),
        "median_abs_gain": float(np.median(diff)),
        "mean_rel_gain_pct": float(np.mean(rel)),
        "median_rel_gain_pct": float(np.median(rel)),
        "max_rel_gain_pct": float(np.max(rel)),
        "wins": int((diff > 0).sum()), "ties": int((abs(diff) < 1e-9).sum()),
        "losses": int((diff < 0).sum()),
        "wilcoxon_p": float(w.pvalue) if w else None,
        "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def build_summary(s3: pd.DataFrame):
    col = _spread_col(s3)
    s3["family"] = s3.graph.map(_family_of)
    s3["act_frac"] = s3[col] / s3["n"]
    out = {}
    for fam in sorted(s3.family.unique()):
        sub = s3[s3.family == fam]
        piv = (sub.groupby(["algorithm"])[col].mean())
        out[f"family_{fam}_means"] = {k: round(float(v), 3) for k, v in piv.items()}
        out[f"family_{fam}_abc_vs_greedy"] = paired_stats(sub, "ABC (proposed)", "Greedy")
        out[f"family_{fam}_abc_vs_celf"] = paired_stats(sub, "ABC (proposed)", "CELF")
        out[f"family_{fam}_abc_vs_sa"] = paired_stats(sub, "ABC (proposed)", "SimulatedAnnealing")
        out[f"family_{fam}_celf_ls_vs_abc"] = paired_stats(sub, "CELF + LS", "ABC (proposed)")

    celf = s3[s3.algorithm == "CELF"]
    chk = celf["x_lazy_bound_checks"].sum() if "x_lazy_bound_checks" in celf else 0
    vio = celf["x_lazy_bound_violations"].sum() if "x_lazy_bound_violations" in celf else 0
    out["celf_lazy_violation_rate_pct"] = round(float(100 * vio / chk), 3) if chk else 0.0

    # absolute scale: mean activation fraction per algorithm
    out["activation_fraction"] = {
        k: round(float(v), 4)
        for k, v in s3.groupby("algorithm")["act_frac"].mean().items()}

    # where ABC beats Greedy in absolute terms
    g = s3[s3.algorithm.isin(["ABC (proposed)", "Greedy"])].groupby(["graph", "k", "algorithm"])[col].mean().unstack()
    g["gain_abs"] = g["ABC (proposed)"] - g["Greedy"]
    g["gain_rel_pct"] = 100 * g["gain_abs"] / g["Greedy"]
    out["abc_vs_greedy_by_config"] = {
        f"{gr}_k{k}": round(float(r.gain_rel_pct), 2)
        for (gr, k), r in g.iterrows()}

    # ---- wall-clock and cascade-success (honest scale indicators) ---- #
    if "runtime_s" in s3.columns:
        out["wall_clock_s_mean"] = {
            k: round(float(v), 2)
            for k, v in s3.groupby("algorithm")["runtime_s"].mean().items()}
    out["cascade_success_rate"] = {
        k: round(float(v), 4)
        for k, v in s3.groupby("algorithm").apply(
            lambda d: (d[col] > d["k"] + 1e-9).mean(), include_groups=False).items()}

    # ---- signed results into the summary ---- #
    signed_path = RESULTS / "s8_signed.csv"
    if signed_path.exists():
        s8 = pd.read_csv(signed_path)
        sp = s8.groupby(["graph", "algorithm"])["spread_heldout"].mean().unstack()
        out["signed_graph_level"] = {
            gr: {a: round(float(v), 3) for a, v in row.dropna().items()}
            for gr, row in sp.iterrows()}
        out["signed_family_means"] = {
            a: round(float(v), 3) for a, v in s8.groupby("algorithm")["spread_heldout"].mean().items()}
        sg = s8[s8.algorithm.isin(["ABC (proposed)", "Greedy"])].groupby(
            ["graph", "k", "algorithm"])["spread_heldout"].mean().unstack()
        out["signed_abc_vs_greedy_pct"] = {
            f"{gr}_k{k}": round(float(100 * (r["ABC (proposed)"] - r["Greedy"]) / max(r["Greedy"], 1e-9)), 2)
            for (gr, k), r in sg.iterrows() if pd.notna(r["ABC (proposed)"]) and pd.notna(r["Greedy"])}

    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    log(json.dumps(out, indent=2, default=str))
    return out


def build_figures():
    log("figures")
    s1 = pd.concat([pd.read_csv(f) for f in sorted(RESULTS.glob("s1_structure*.csv"))],
                   ignore_index=True)
    s1.to_csv(RESULTS / "s1_all.csv", index=False)
    s2 = pd.read_csv(RESULTS / "s2_exact.csv")
    parts = [pd.read_csv(f) for f in sorted(RESULTS.glob("s3_main_*.csv"))]
    s3 = pd.concat(parts, ignore_index=True)
    s3.to_csv(RESULTS / "s3_main.csv", index=False)
    s5_files = sorted(RESULTS.glob("s5_delta_*.csv"))
    curves = []
    for f in sorted(RESULTS.glob("s4_curves_*.json")):
        curves.extend(json.loads(f.read_text()))

    fig_structure(s1)
    fig_spread_vs_k(s3)
    fig_convergence(curves)
    for f in s5_files:
        gname = f.stem.replace("s5_delta_", "")
        fig_delta(pd.read_csv(f), gname)
    fig_exact(s2)
    if (RESULTS / "s9_budget.csv").exists():
        fig_budget(pd.read_csv(RESULTS / "s9_budget.csv"))

    build_summary(s3)
    col = _spread_col(s3)
    piv = (s3.groupby(["graph", "k", "algorithm"])[col].mean().unstack())
    piv.to_csv(RESULTS / "summary_pivot.csv")


def main():
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage.startswith("s1:"):
        study_structure(only=stage.split(":", 1)[1])
    elif stage in ("s1", "all"):
        study_structure()
    if stage in ("s2", "all"):
        study_exact()
    if stage.startswith("s3:"):
        spec = stage.split(":", 1)[1]
        g, _, kspec = spec.partition("@")
        ks = tuple(int(x) for x in kspec.split(",")) if kspec else (5, 10, 20, 30)
        tag = f"{g}_{kspec.replace(',', '-')}" if kspec else g
        df, curves = study_main(graphs=(g,), ks=ks)
        df.to_csv(RESULTS / f"s3_main_{tag}.csv", index=False)
        (RESULTS / f"s4_curves_{tag}.json").write_text(json.dumps(curves))
    if stage in ("s5", "all"):
        study_delta(gname="ba500")
    if stage == "s5:ws":
        study_delta(gname="ws1000")
    if stage in ("s8", "all"):
        study_signed()
    if stage in ("s9", "all"):
        study_budget()
    if stage in ("s10", "all"):
        study_multi()
    if stage in ("s11", "all"):
        study_sign_control()
    if stage in ("fig", "all"):
        build_figures()


if __name__ == "__main__":
    main()