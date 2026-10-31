r"""
baselines.py
============
Set-level baselines for the equal-budget comparison, sharing the same swap
moves and the same sigma / potential oracle as ABC so that the comparison
isolates the SEARCH STRATEGY, not the representation.

    * simulated_annealing : 1-swap neighbourhood, fitness = sigma + lam*Phi,
      geometric cooling, within the evaluation budget.
    * grasp               : GRASP = greedy-randomised construction (random
      seeds refined by first-improvement local search) restarted within the
      budget.
"""

from __future__ import annotations

import time

import numpy as np

from majority_rule_model import MajorityRuleModel, SelectionResult, celf


def _fitness_of(model, s, lam):
    if lam > 0.0:
        sg, ph, _se = model.sigma_potential(s)
        return sg, sg + lam * ph
    v = model.sigma(s)
    return v, v


def simulated_annealing(
    model: MajorityRuleModel,
    k: int,
    pool,
    eval_budget: int,
    lam: float = 0.05,
    seed: int = 0,
    restart: bool = False,
    name: str = "SimulatedAnnealing",
) -> SelectionResult:
    """
    Simulated annealing over size-k sets with 1-swap moves and geometric
    cooling.  The acceptance uses the potential-smoothed fitness; the
    incumbent is tracked on sigma alone.  `restart=True` re-heats once when
    the temperature is exhausted and budget remains.
    """
    t0, e0 = time.perf_counter(), model.n_evals
    rng = np.random.default_rng(seed)
    pool = list(pool)

    def spent() -> int:
        return model.n_evals - e0

    def budget_left() -> bool:
        return spent() < eval_budget

    S = rng.choice(pool, size=k, replace=False).tolist()
    sg, f = _fitness_of(model, S, lam)
    best_set, best_sg = list(S), sg
    T0, T1 = max(0.05 * sg, 0.5), 0.01
    iters = max(10, eval_budget)
    loga = np.log(T0 / T1) / iters
    T = T0

    def run():
        nonlocal S, sg, f, best_set, best_sg, T
        stall = 0
        while budget_left() and stall < 1500:
            cand = list(S)
            pos = int(rng.integers(k))
            out = cand[pos]
            options = [v for v in pool if v not in cand]
            cand[pos] = options[int(rng.integers(len(options)))]
            nsg, nf = _fitness_of(model, cand, lam)
            delta = nf - f
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                S, sg, f = cand, nsg, nf
                stall = 0
                if nsg > best_sg:
                    best_set, best_sg = list(S), nsg
            else:
                stall += 1
            T *= np.exp(-loga)
        if restart:
            T = T0

    run()
    if restart and budget_left():
        run()
    return SelectionResult(
        algorithm=name, seeds=list(best_set), spread=best_sg,
        runtime=time.perf_counter() - t0, evaluations=spent(),
        extra={"temp_start": T0, "restart": restart},
    )


def grasp(
    model: MajorityRuleModel,
    k: int,
    pool,
    eval_budget: int,
    lam: float = 0.05,
    seed: int = 0,
    name: str = "GRASP",
) -> SelectionResult:
    """
    GRASP: random greedy construction + first-improvement 1-opt, restarted
    until the budget is gone.  Construction picks a random node among the top
    3 by sigma gain each round (greedy-randomised).
    """
    t0, e0 = time.perf_counter(), model.n_evals
    rng = np.random.default_rng(seed)
    pool = list(pool)
    best_set, best_sg = None, -np.inf

    def spent() -> int:
        return model.n_evals - e0

    def budget_left() -> bool:
        return spent() < eval_budget

    while budget_left():
        S: list[int] = []
        for _ in range(k):
            gains = []
            for v in pool:
                if v not in S and budget_left():
                    sg, _ = _fitness_of(model, S + [v], lam)
                    gains.append((sg, v))
            if not gains:
                break
            gains.sort(reverse=True)
            top = gains[: min(3, len(gains))]
            _, v = top[int(rng.integers(len(top)))]
            S.append(v)
        # first-improvement local search
        improved = True
        while improved and budget_left():
            improved = False
            for pos in range(k):
                if not budget_left():
                    break
                cand = list(S)
                options = [v for v in pool if v not in cand]
                cand[pos] = options[int(rng.integers(len(options)))]
                if model.sigma(cand) > model.sigma(S) + 1e-9:
                    S = cand
                    improved = True
        sg = model.sigma(S)
        if sg > best_sg:
            best_set, best_sg = list(S), sg
    return SelectionResult(
        algorithm=name, seeds=list(best_set), spread=best_sg,
        runtime=time.perf_counter() - t0, evaluations=spent(),
    )


def sandwich_celf(
    model: MajorityRuleModel,
    k: int,
    pool,
    name: str = "Sandwich-CELF",
) -> SelectionResult:
    """
    CELF on the sandwich upper-bound model sigma_bar (variant "slt"): the
    concave envelope of the majority CDF, pathwise >= sigma and submodular,
    so CELF is VALID here.  Returns the seed set scored on the ORIGINAL model.
    """
    t0 = time.perf_counter()
    bar = type(model)(model.graph, variant="slt", delta=model.delta,
                      n_samples=model.R, seed=model.seed,
                      backend=model.backend)
    r_bar = celf(bar, k, pool=pool)
    spread = model.sigma(r_bar.seeds)
    return SelectionResult(
        algorithm=name, seeds=list(r_bar.seeds), spread=spread,
        runtime=time.perf_counter() - t0,
        evaluations=r_bar.evaluations,
        extra={"bar_spread": r_bar.spread,
               "bar_ratio": float(bar.sigma(r_bar.seeds) / max(spread, 1e-9)),
               "bar_opt_bound": float(r_bar.spread * np.e / (np.e - 1.0)),
               "alpha": float(bar.sigma(r_bar.seeds) / max(spread, 1e-9))},
    )


if __name__ == "__main__":
    from majority_rule_model import load_graph, candidate_pool, celf as _c
    G = load_graph("ba500")
    m = MajorityRuleModel(G, "smr", 0.15, 150, seed=7)
    pool = candidate_pool(m, 200)
    budget = 10 * len(pool)
    for fn in (simulated_annealing, grasp, sandwich_celf):
        m.reset_counters(); m.clear_cache()
        r = fn(m, 10, pool, eval_budget=budget)
        print(f"{r.algorithm:20s} sigma={r.spread:8.3f} evals={r.evaluations}")
        if r.extra:
            print("   extra:", {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in r.extra.items()})