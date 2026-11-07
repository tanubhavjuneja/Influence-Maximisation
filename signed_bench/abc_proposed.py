"""
abc_proposed.py
===============
Self-contained port of the project's optimised Artificial Bee Colony
(`abc_celf.py` in the parent directory), re-written with numpy ONLY
(no scipy/networkx).  Used in this benchmark as "ABC (proposed)".

The configuration is EXACTLY the one used in the real-network runs:

    ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
              seed_with_celf=True, eval_budget=k*200,
              random_state=600 + restart)

Design recap (see ABC_PROPOSED_CODE_DOC.md in the parent directory):
  * CELF warm start (lazy greedy, audited),
  * discrete ABC over whole k-sets: gbest-guided partners, tournament
    onlookers, directed scouts, duplicate suppression,
  * fitness = sigma + lambda * Phi (plateau gradient), lexicographic
    accept rule with Monte-Carlo-noise tolerance,
  * budget-respecting local refinement (disabled here: refine_every=0).

Omitted from the port (unused by the benchmark config, documented for
honesty): sandwich seeding (needs a second model class) and
seed_with_exchange.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectionResult:
    algorithm: str
    seeds: list[int]
    spread: float
    runtime: float
    evaluations: int
    trace: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  CELF (lazy forward greedy with an audited stale bound)
# --------------------------------------------------------------------------- #
def celf(model, k: int, pool: list[int]) -> SelectionResult:
    """Port of majority_rule_model.celf.  Under the (non-submodular) signed
    rule the stale bound is a HEURISTIC; violations are counted."""
    t0, e0 = time.perf_counter(), model.n_evals
    cand = list(pool)
    base = model.sigma([])
    heap: list[tuple[float, int, int]] = []
    stale: dict[int, float] = {}
    for v in cand:
        g = model.sigma([v]) - base
        stale[v] = g
        heapq.heappush(heap, (-g, v, 0))
    S: list[int] = []
    cur = base
    violations = checks = 0
    while len(S) < k and heap:
        neg_g, v, last_round = heapq.heappop(heap)
        if last_round == len(S):
            S.append(v)
            cur += -neg_g
            continue
        new_g = model.sigma(S + [v]) - cur
        checks += 1
        if new_g > stale[v] + 1e-9:
            violations += 1
        stale[v] = new_g
        heapq.heappush(heap, (-new_g, v, len(S)))
    return SelectionResult(
        "CELF", S, cur, time.perf_counter() - t0,
        model.n_evals - e0,
        extra={"lazy_bound_checks": checks,
               "lazy_bound_violations": violations,
               "lazy_violation_rate": (violations / checks if checks else 0.0)})


# --------------------------------------------------------------------------- #
#  Local refinement (lazy 1-opt + 2-swap) -- unused at refine_every=0 but
#  ported for completeness
# --------------------------------------------------------------------------- #
def _cascade_cands(model, S_minus, cands, cap):
    relevant = set()
    for v in S_minus:
        relevant.update(model.neighbours(v).tolist())
    for b in model.diagnostics(S_minus, 30):
        relevant.update(model.neighbours(b).tolist())
    relevant.difference_update(S_minus)
    out = [v for v in cands if v in relevant]
    if len(out) < 4:
        out = list(cands)
    return out[:cap]


def celf_exchange_refine(model, seeds, pool, max_passes=3, eval_budget=None,
                         lazy=True, audit=True, cascade_aware=True,
                         max_cands=60, two_swap=True, two_swap_tries=400):
    S = list(dict.fromkeys(seeds))
    best = model.sigma(S)
    start_evals = model.n_evals
    violations = checks = swaps = 0
    base_cache_key = frozenset()

    def spent():
        return model.n_evals - start_evals

    def over():
        return eval_budget is not None and spent() >= eval_budget

    for _ in range(max_passes):
        if over():
            break
        improved = False
        for pos in range(len(S)):
            if over():
                break
            out = S[pos]
            S_minus = S[:pos] + S[pos + 1:]
            base = model.sigma(S_minus)
            base_key = frozenset(S_minus)
            cands = [v for v in pool if v not in S_minus]
            if cascade_aware and len(cands) > max_cands:
                cands = _cascade_cands(model, S_minus, cands, max_cands)
            if not cands:
                continue
            if lazy:
                _SINGLETON = object()
                heap: list[tuple[float, int, object]] = []
                for v in cands:
                    s0 = model.cache.get(base_cache_key | {v})
                    e0_ = model.cache.get(base_cache_key)
                    if s0 is not None and e0_ is not None:
                        heap.append((-(s0 - e0_), v, _SINGLETON))
                    else:
                        heap.append((np.inf, v, None))
                heapq.heapify(heap)
                best_v, best_val = None, best
                while heap:
                    neg, v, key = heapq.heappop(heap)
                    if key == base_key:
                        best_v, best_val = v, base + (-neg)
                        break
                    if over():
                        break
                    gain = model.sigma(S_minus + [v]) - base
                    if audit and key is not None:
                        checks += 1
                        if gain > -neg + 1e-9:
                            violations += 1
                    heapq.heappush(heap, (-gain, v, base_key))
            else:
                if over():
                    break
                vals = {v: model.sigma(S_minus + [v]) for v in cands}
                best_v = max(vals, key=vals.get)
                best_val = vals[best_v]
            if best_v is not None and best_val > best + 1e-9 and best_v != out:
                S = S_minus + [best_v]
                best = best_val
                swaps += 1
                improved = True
        if not improved:
            break

    pair_swaps = 0
    if two_swap and not over() and len(pool) > len(S) + 1:
        rng = np.random.default_rng(7)
        for _ in range(2):
            if over():
                break
            improved = False
            for _ in range(two_swap_tries):
                if over():
                    break
                out_pair = set(rng.choice(len(S), size=2, replace=False).tolist())
                outside = [v for v in pool if v not in S]
                ins_pair = rng.choice(len(outside), size=2, replace=False).tolist()
                S2 = [v for v in S if v not in {S[i] for i in out_pair}]
                S2 = S2 + [outside[i] for i in ins_pair]
                val = model.sigma(S2)
                if val > best + 1e-9:
                    S, best = S2, val
                    swaps += 1
                    pair_swaps += 1
                    improved = True
                    break
            if not improved:
                break

    return S, best, {"refine_swaps": swaps,
                     "refine_pair_swaps": pair_swaps,
                     "refine_evals": spent(),
                     "lazy_bound_checks": checks,
                     "lazy_bound_violations": violations}


# --------------------------------------------------------------------------- #
#  Discrete ABC
# --------------------------------------------------------------------------- #
@dataclass
class ABCConfig:
    colony_size: int = 30
    max_cycles: int = 80
    limit: int = 6
    mutation_rate: float = 0.2
    n_mod: int = 1
    gbest_prob: float = 0.6
    selection: str = "tournament"
    tournament_size: int = 4
    directed_scout: bool = True
    dedupe: bool = True
    accept_equal: bool = True
    explore_prob: float = 0.15
    potential_weight: float = 0.05
    tol_mult: float = 2.0
    accept_rule: str = "lex"
    temp_frac: float = 0.01
    temp_min: float = 0.25
    temp_decay: float = 0.97
    refine_every: int = 0
    refine_passes: int = 2
    lazy_refine: bool = True
    two_swap: bool = True
    cascade_aware: bool = True
    seed_mode: str = "celf"
    seed_with_celf: bool = True
    adaptive_colony: bool = False
    reserve_fraction: float = 0.15
    stagnation_limit: int = 0
    eval_budget: int | None = None
    random_state: int = 0
    verbose: bool = False


@dataclass
class ABCTrace:
    cycle: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    evals: list[int] = field(default_factory=list)


def _biased_set(model, pool, k, rng):
    w = np.asarray(model.pool_weight + 1.0, dtype=float)
    w = w[np.asarray(pool, dtype=int)]
    w = w / w.sum()
    return rng.choice(pool, size=k, replace=False, p=w).tolist()


def _neighbour_solution(x, partner, pool, k, rng, cfg):
    partner_set = set(partner)
    xs = list(dict.fromkeys(x))
    for _ in range(max(1, cfg.n_mod)):
        only_i = [v for v in xs if v not in partner_set]
        only_j = [v for v in partner_set if v not in xs]
        if only_i and only_j:
            drop = only_i[rng.integers(len(only_i))]
            take = only_j[rng.integers(len(only_j))]
            xs[xs.index(drop)] = take
        else:
            drop = xs[rng.integers(len(xs))]
            options = [v for v in pool if v not in xs]
            if options:
                xs[xs.index(drop)] = options[rng.integers(len(options))]
        if rng.random() < cfg.mutation_rate:
            cur = set(xs)
            options = [v for v in pool if v not in cur]
            if options:
                drop = xs[rng.integers(len(xs))]
                xs[xs.index(drop)] = options[rng.integers(len(options))]
    return list(dict.fromkeys(xs))[:k]


def _candidate(x, partner, pool, k, rng, cfg):
    if rng.random() < cfg.explore_prob:
        return rng.choice(pool, size=k, replace=False).tolist()
    return _neighbour_solution(x, partner, pool, k, rng, cfg)


def _pick_sources(fitness, rng, cfg):
    n = len(fitness)
    fit = np.asarray(fitness, dtype=float)
    if cfg.selection == "roulette":
        shifted = fit - fit.min() + 1e-6
        probs = shifted / shifted.sum()
        return rng.choice(n, size=n, p=probs)
    picks = rng.integers(n, size=(n, max(2, cfg.tournament_size)))
    winners = picks[np.arange(n), np.argmax(fit[picks], axis=1)]
    return winners


def abc_optimize(model, k, pool, cfg=None, warm_start=None,
                 name="ABC (proposed)") -> SelectionResult:
    cfg = cfg or ABCConfig()
    rng = np.random.default_rng(cfg.random_state)
    t0, e0 = time.perf_counter(), model.n_evals
    pool = list(pool)
    n_sources = max(4, cfg.colony_size // 2)
    if cfg.adaptive_colony:
        n_sources = max(4, min(n_sources, len(pool) // max(1, k)))
    reserve = (cfg.reserve_fraction if cfg.refine_every else 0.0)
    soft_budget = (cfg.eval_budget * (1.0 - reserve)) if cfg.eval_budget else None

    def spent():
        return model.n_evals - e0

    def budget_left():
        return soft_budget is None or spent() < soft_budget

    def remaining():
        if cfg.eval_budget is None:
            return None
        return max(0, cfg.eval_budget - spent())

    # ---- initialisation ----
    population: list[list[int]] = []
    celf_stats: dict = {}
    if cfg.seed_with_celf and cfg.seed_mode == "celf":
        r = celf(model, k, pool=pool)
        population.append(list(r.seeds))
        celf_stats = {"celf_seed_spread": r.spread}
    if warm_start:
        for w in warm_start:
            if len(w) == k:
                population.append(list(w))
    while len(population) < n_sources:
        population.append(_biased_set(model, pool, k, rng))
    population = population[:n_sources]

    def fitness_of(s):
        if cfg.potential_weight > 0.0:
            sg, ph, se = model.sigma_potential(s)
            return sg, sg + cfg.potential_weight * ph, se
        v = model.sigma(s)
        return v, v, 0.0

    fit_pairs = [fitness_of(s) for s in population]
    fitness = [f for _, f, _ in fit_pairs]
    trials = [0] * n_sources

    best_i = int(np.argmax([s for s, _, _ in fit_pairs]))
    best_set, best_fit = list(population[best_i]), fit_pairs[best_i][0]

    trace = ABCTrace()
    refine_stats = {"refine_swaps": 0, "refine_pair_swaps": 0,
                    "lazy_bound_checks": 0, "lazy_bound_violations": 0}
    scout_resets = dup_replacements = 0
    last_gain_cycle = 0
    refine_active = True
    zero_refines = 0
    temp = [max(cfg.temp_frac * best_fit, cfg.temp_min)]

    def accept(ns, nph, nse, i):
        cs, cph, cse = fit_pairs[i]
        tol = cfg.tol_mult * max(cse, nse)
        if ns > cs + tol:
            return True
        if abs(ns - cs) <= tol:
            return nph > cph + 1e-9 or (cfg.accept_equal
                                        and abs(nph - cph) <= 1e-9)
        if cfg.accept_rule == "metro":
            return rng.random() < float(np.exp((ns - cs) / max(temp[0], 1e-9)))
        return False

    def dedupe():
        nonlocal dup_replacements
        seen: set[frozenset] = set()
        for idx in range(n_sources):
            key = frozenset(population[idx])
            if key in seen:
                population[idx] = _biased_set(model, pool, k, rng)
                fit_pairs[idx] = fitness_of(population[idx])
                fitness[idx] = fit_pairs[idx][1]
                trials[idx] = 0
                dup_replacements += 1
            seen.add(key)

    # ---- main loop ----
    for cycle in range(1, cfg.max_cycles + 1):
        if not budget_left():
            break
        if cfg.stagnation_limit and cycle - last_gain_cycle > cfg.stagnation_limit:
            break

        # employed bees
        for i in range(n_sources):
            if not budget_left():
                break
            if rng.random() < cfg.gbest_prob and len(population) > 1:
                partner = best_set
            else:
                j = int(rng.integers(n_sources))
                while j == i and n_sources > 1:
                    j = int(rng.integers(n_sources))
                partner = population[j]
            cand = _candidate(population[i], partner, pool, k, rng, cfg)
            ns, nph, nse = fitness_of(cand)
            if accept(ns, nph, nse, i):
                population[i], trials[i] = cand, 0
                fit_pairs[i] = (ns, nph, nse)
                fitness[i] = ns + cfg.potential_weight * nph
            else:
                trials[i] += 1

        # onlooker bees
        sources = _pick_sources(fitness, rng, cfg)
        for i in sources:
            if not budget_left():
                break
            if rng.random() < cfg.gbest_prob:
                partner = best_set
            else:
                j = int(rng.integers(n_sources))
                while j == i and n_sources > 1:
                    j = int(rng.integers(n_sources))
                partner = population[j]
            cand = _candidate(population[i], partner, pool, k, rng, cfg)
            ns, nph, nse = fitness_of(cand)
            if accept(ns, nph, nse, i):
                population[i], trials[i] = cand, 0
                fit_pairs[i] = (ns, nph, nse)
                fitness[i] = ns + cfg.potential_weight * nph
            else:
                trials[i] += 1

        # scout bees
        for i in range(n_sources):
            if trials[i] > cfg.limit and budget_left():
                if cfg.directed_scout and rng.random() < 0.7:
                    cand = _neighbour_solution(
                        best_set, best_set, pool, k, rng,
                        ABCConfig(n_mod=2, mutation_rate=1.0, gbest_prob=0.0))
                else:
                    cand = rng.choice(pool, size=k, replace=False).tolist()
                population[i] = cand
                fit_pairs[i] = fitness_of(cand)
                fitness[i] = fit_pairs[i][1]
                trials[i] = 0
                scout_resets += 1

        if cfg.dedupe:
            dedupe()

        # elitism
        i = int(np.argmax([s for s, _, _ in fit_pairs]))
        if fit_pairs[i][0] > best_fit:
            best_fit, best_set = fit_pairs[i][0], list(population[i])
            last_gain_cycle = cycle

        # local refinement of the incumbent (refine_every=0 -> never runs)
        if cfg.refine_every and refine_active and cycle % cfg.refine_every == 0 \
                and budget_left():
            S2, f2, st = celf_exchange_refine(
                model, best_set, pool, max_passes=cfg.refine_passes,
                eval_budget=remaining(), lazy=cfg.lazy_refine,
                two_swap=cfg.two_swap, cascade_aware=cfg.cascade_aware)
            for key in ("refine_swaps", "refine_pair_swaps",
                        "lazy_bound_checks", "lazy_bound_violations"):
                refine_stats[key] += st[key]
            if f2 > best_fit:
                best_fit, best_set = f2, list(S2)
                last_gain_cycle = cycle
                worst = int(np.argmin(fitness))
                population[worst], trials[worst] = list(S2), 0
                fit_pairs[worst] = fitness_of(S2)
                fitness[worst] = fit_pairs[worst][1]
            if st["refine_swaps"] == 0:
                zero_refines += 1
                if zero_refines >= 2:
                    refine_active = False

        trace.cycle.append(cycle)
        trace.best.append(best_fit)
        trace.mean.append(float(np.mean([s for s, _, _ in fit_pairs])))
        trace.evals.append(spent())
        if cfg.accept_rule == "metro":
            temp[0] = max(temp[0] * cfg.temp_decay, cfg.temp_min)
            temp[0] = max(temp[0], cfg.temp_frac * best_fit)
        if cfg.verbose and cycle % 10 == 0:
            print(f"  cycle {cycle:3d}  best={best_fit:8.3f}  "
                  f"mean={np.mean([s for s, _, _ in fit_pairs]):8.3f}  "
                  f"evals={spent()}")

    # ---- final polish (within budget) ----
    if cfg.refine_every and refine_active and \
            (cfg.eval_budget is None or remaining() > 0):
        S2, f2, st = celf_exchange_refine(
            model, best_set, pool, max_passes=cfg.refine_passes + 1,
            eval_budget=remaining(), lazy=cfg.lazy_refine,
            two_swap=cfg.two_swap, cascade_aware=cfg.cascade_aware)
        for key in ("refine_swaps", "refine_pair_swaps",
                    "lazy_bound_checks", "lazy_bound_violations"):
            refine_stats[key] += st[key]
        if f2 > best_fit:
            best_fit, best_set = f2, list(S2)

    return SelectionResult(
        algorithm=name,
        seeds=list(best_set),
        spread=best_fit,
        runtime=time.perf_counter() - t0,
        evaluations=spent(),
        trace=trace.best,
        extra={"cycles_run": len(trace.cycle),
               "scout_resets": scout_resets,
               "dup_replacements": dup_replacements,
               **refine_stats,
               **celf_stats})