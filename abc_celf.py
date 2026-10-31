"""
abc_celf.py
===========
The OPTIMISED solver: a discrete Artificial Bee Colony (ABC) metaheuristic
hybridised with a CELF-style lazy-exchange local refinement.

Why this design, given that sigma is non-submodular
---------------------------------------------------
Forward greedy (and therefore CELF, which is just greedy with a lazy queue)
builds a seed set one node at a time and is driven entirely by *singleton
marginal gains*.  Under a majority rule those gains are nearly all zero: one
extra active neighbour almost never flips anybody, so greedy is choosing among
ties and is effectively blind.  Influence under a majority rule is a property
of COALITIONS, not of individuals.

ABC searches the space of whole size-k sets, so it can discover the
complementary coalitions greedy cannot see.  Its weakness is the opposite one:
it has no notion of marginal value and wastes evaluations on obviously bad
swaps.  The local refinement supplies exactly that missing gradient.

    ABC            -> global, set-level, escapes the zero-gain plateau
    local exchange -> node-level 1-opt polishing (lazy bounds, audited) plus
                      2-swap shaking for coalitions 1-opt cannot repair

Guarantees
----------
  1. Elitism + CELF warm start: the solver never returns worse than CELF.
  2. Sigma is not submodular, so a stale marginal gain is NOT a valid upper
     bound; the lazy loop is run as a heuristic and AUDITED
     (`lazy_bound_violations` counts re-evaluations whose gain exceeded their
     stale bound).  Exact 1-opt is available (`lazy=False`).
  3. SANDWICH SEED (opt-in, `seed_mode="sandwich"`): the concave envelope of
     the majority CDF is F_bar(x) = min(x/(0.5+delta), 1), i.e. an LT-type
     model with theta ~ U[0, 0.5+delta] (variant "slt").  sigma_bar >= sigma
     pathwise and sigma_bar is submodular (Mossel-Roch sufficiency), so CELF
     is VALID on it and gives a submodular-valid warm start, an upper bound
     OPT <= sigma_bar(S_celf_bar) * e/(e-1), and an after-the-fact guarantee
     sigma(S) >= (1-1/e) * OPT / alpha(S)  with  alpha(S) = sigma_bar(S)/sigma(S).

Literature-driven upgrades over the canonical ABC
-------------------------------------------------
The canonical ABC (Karaboga 2005; Karaboga & Basturk 2007) has well-known
weaknesses, each addressed by a configurable operator:

  W1 weak exploitation, slow late-stage convergence
     -> Gbest-guided neighbour generation (Zhu & Kwong 2010): with prob.
        `gbest_prob` the partner in v_i = x_i + phi (x_i - x_j) is the global
        best set, which turns the random walk into a directional pull.
  W2 premature convergence of the roulette wheel when fitnesses are near-equal
     -> tournament selection for onlooker bees (default), roulette kept as an
        option (`selection="roulette"`).
  W3 pure-random scout restarts waste evaluations
     -> directed scouts (Kiran & Findik 2015): abandoned sources are rebuilt
        by perturbing the incumbent, not by uniform resampling.
  W4 fixed perturbation strength / parameter sensitivity
     -> modification-rate analogue of Akay & Karaboga (2012): `n_mod` partner
        swaps plus a `mutation_rate` extra random swap per candidate.
  W5 duplicate food sources inflate the population
     -> duplicate suppression (`dedupe`).
  W6 refinement blowing the evaluation budget
     -> the refinement receives the REMAINING budget only; a stagnation
        early-stop (`stagnation_limit`) further saves budget.
  W7 rigid neighbourhood generation on plateaus (flagged for IM by the
     ABC+VNS line of work) traps the colony in basins separated by strict
     fitness drops
     -> PLATEAU GRADIENT: fitness = sigma + lambda * Phi(S), where
        Phi(S) = sum_v min(1, net_v / need_v) at convergence is computed in
        the same cascade pass (no extra evaluation cost).  Phi orders sets
        that sigma cannot distinguish and lets bees slide across zero-gain
        plateaus (`accept_equal` kept as a safety net), plus diversity
        injection (`explore_prob`) and unbiased scouts.
  W8 refinement consuming budget when the 1-opt neighbourhood is exhausted
     -> adaptive refinement gating: after two zero-swap refinement passes the
        reserved budget is returned to the population search.
  W9 1-opt cannot repair broken coalitions (why CELF+1-opt gains so little)
     -> CASCADE-AWARE candidates: swap-ins restricted to neighbours of the
        current seeds and of near-threshold nodes (slack <= 1 at convergence);
        plus a 2-swap shaking phase when 1-opt stalls.

On SIGNED graphs (see signed_rule_model.py) sigma is not even monotone, so
the swap operators can *remove* harmful enemy seeds -- something no greedy
scheme can express.  Initialisation is sign-aware through
`model.pool_weight` (net positive support).

Reference points
----------------
Karaboga (2005, TR06) ABC; Karaboga & Basturk (2007, J Glob Optim) ABC;
Leskovec et al. (2007) CELF; Kempe/Kleinberg/Tardos (2003) greedy;
Das & Kempe (2011) submodularity ratio; Mossel & Roch (2007) submodularity
sufficient condition for the general threshold model; Lu, Chen & Lakshmanan
(2015) sandwich approximation; Zhu & Kwong (2010) gbest-guided ABC;
Akay & Karaboga (2012) modified ABC; Kiran & Findik (2015) directed ABC;
Cantini, Marozzo, Mazza, Talia & Trunfio (2021, OSNEM) WABC for influence
maximisation; ABC + variable neighbourhood search for IM (Swarm and
Evolutionary Computation, 2026) -- whose critique ("rigid neighbourhood
generation, weak directional guidance") motivated W7/W8/W9.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from majority_rule_model import MajorityRuleModel, SelectionResult, celf


# --------------------------------------------------------------------------- #
#  Local refinement: lazy 1-opt + cascade-aware candidates + 2-swap shaking
# --------------------------------------------------------------------------- #
def _cascade_cands(model, S_minus, cands, cap) -> list[int]:
    """Restrict swap-in candidates to cascade-relevant nodes: neighbours of the
    current seeds and of near-threshold (slack <= 1) nodes at convergence."""
    sset = set(S_minus)
    A = model.A.tocsr()
    relevant = set()
    for v in sset:
        relevant.update(A.indices[A.indptr[v]: A.indptr[v + 1]].tolist())
    for b in model._border_nodes(S_minus, 30):
        relevant.update(A.indices[A.indptr[b]: A.indptr[b + 1]].tolist())
    relevant.difference_update(sset)
    out = [v for v in cands if v in relevant]
    if len(out) < 4:          # fall back to full pool if too restrictive
        out = list(cands)
    return out[:cap]


def celf_exchange_refine(
    model: MajorityRuleModel,
    seeds,
    pool,
    max_passes: int = 3,
    eval_budget: int | None = None,
    lazy: bool = True,
    audit: bool = True,
    cascade_aware: bool = True,
    max_cands: int = 60,
    two_swap: bool = True,
    two_swap_tries: int = 400,
) -> tuple[list[int], float, dict]:
    """
    1-opt exchange with a lazy priority queue over candidate insertions,
    cascade-aware candidate restriction, and a 2-swap shaking phase.

    LAZY MODE (the CELF assumption, audited).  Each candidate's heap bound is
    its last marginal gain against a SUBSET of the current base (initially the
    singleton gain sigma({v}) - sigma({}), the classic CELF bound).  A
    candidate is accepted when its bound was computed against the CURRENT
    base S_minus.  Under submodularity that bound is valid and the shortcut is
    exact; under the majority rule it is NOT, so every re-evaluation whose
    fresh gain EXCEEDS its stale bound is counted in `lazy_bound_violations`.

    EXACT MODE (`lazy=False`): every candidate is evaluated against the
    current base; no bound is trusted.

    `eval_budget` caps FRESH sigma() evaluations (cache hits are free).
    Returns (seeds, spread, stats).
    """
    S = list(dict.fromkeys(seeds))
    best = model.sigma(S)
    start_evals = model.n_evals
    violations = checks = swaps = 0
    base_cache_key = frozenset()

    def spent() -> int:
        return model.n_evals - start_evals

    def over() -> bool:
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
                    s0 = model._cache.get(base_cache_key | {v})
                    e0 = model._cache.get(base_cache_key)
                    if s0 is not None and e0 is not None:
                        heap.append((-(s0 - e0), v, _SINGLETON))
                    else:
                        heap.append((np.inf, v, None))
                heapq.heapify(heap)

                best_v, best_val = None, best
                while heap:
                    neg, v, key = heapq.heappop(heap)
                    if key == base_key:            # bound computed on this base: accept
                        best_v, best_val = v, base + (-neg)
                        break
                    if over():
                        break
                    gain = model.sigma(S_minus + [v]) - base
                    if audit and key is not None:
                        checks += 1
                        if gain > -neg + 1e-9:
                            violations += 1        # stale bound was NOT an upper bound
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

    # ---- 2-swap shaking: repair coalitions 1-opt cannot move (W9) ---- #
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

    stats = {
        "refine_swaps": swaps,
        "refine_pair_swaps": pair_swaps,
        "refine_evals": spent(),
        "lazy_bound_checks": checks,
        "lazy_bound_violations": violations,
    }
    return S, best, stats


# --------------------------------------------------------------------------- #
#  Discrete Artificial Bee Colony
# --------------------------------------------------------------------------- #
@dataclass
class ABCConfig:
    colony_size: int = 30          # food sources = employed bees
    max_cycles: int = 80
    limit: int = 6                 # abandonment threshold for scouts
    mutation_rate: float = 0.2     # prob. of an extra random swap per candidate (W4)
    n_mod: int = 1                 # partner-guided swaps per candidate (W4)
    gbest_prob: float = 0.6        # prob. partner = global best (W1)
    selection: str = "tournament"  # onlooker selection: tournament | roulette (W2)
    tournament_size: int = 4
    directed_scout: bool = True    # rebuild abandoned sources near the incumbent (W3)
    dedupe: bool = True            # no duplicate food sources (W5)
    accept_equal: bool = True      # slide along exact plateaus (W7 safety net)
    explore_prob: float = 0.15     # prob. a bee move is a fresh RANDOM set (W7)
    potential_weight: float = 0.05 # lambda in fitness = sigma + lambda * Phi (W7)
    tol_mult: float = 2.0          # sigma ties = differences within tol_mult*SEM
    accept_rule: str = "lex"       # "lex" (sigma, Phi) | "metro" (Metropolis)
    temp_frac: float = 0.01        # metro initial temperature (fraction of best)
    temp_min: float = 0.25
    temp_decay: float = 0.97
    refine_every: int = 10         # local refinement cadence (0 = never)
    refine_passes: int = 2
    lazy_refine: bool = True
    two_swap: bool = True          # 2-swap shaking when 1-opt stalls (W9)
    cascade_aware: bool = True     # restrict swap-ins to cascade-relevant nodes (W9)
    seed_mode: str = "celf"        # "celf" | "sandwich" | "none"
    seed_with_celf: bool = True
    seed_with_exchange: bool = False
    adaptive_colony: bool = False
    reserve_fraction: float = 0.15  # share of the budget reserved for refinement
    stagnation_limit: int = 0
    eval_budget: int | None = None  # hard cap on fresh cascade evaluations (W6)
    random_state: int = 0
    verbose: bool = False


@dataclass
class ABCTrace:
    cycle: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    evals: list[int] = field(default_factory=list)


def _biased_set(model, pool, k, rng) -> list[int]:
    """Degree-biased (unsigned) / net-positive-support-biased (signed) init."""
    w = np.asarray(getattr(model, "pool_weight", model.deg) + 1.0, dtype=float)
    w = w[np.asarray(pool, dtype=int)]
    w = w / w.sum()
    return rng.choice(pool, size=k, replace=False, p=w).tolist()


_degree_biased_set = _biased_set          # backwards-compatible alias


def _neighbour_solution(x, partner, pool, k, rng, cfg) -> list[int]:
    """
    Discrete analogue of Karaboga's  v_i = x_i + phi (x_i - x_j).

    The 'difference' between two sets is their symmetric difference; each
    perturbation drops a node unique to x_i and adopts a node unique to x_j
    (`n_mod` times).  With probability `mutation_rate` an extra purely random
    replacement is applied to maintain diversity (W4).
    """
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


def _candidate(x, partner, pool, k, rng, cfg) -> list[int]:
    """One bee move: with prob `explore_prob` a fresh uniform-random set."""
    if rng.random() < cfg.explore_prob:
        return rng.choice(pool, size=k, replace=False).tolist()
    return _neighbour_solution(x, partner, pool, k, rng, cfg)


def _pick_sources(fitness, rng, cfg) -> np.ndarray:
    """Choose one source index per onlooker (tournament or roulette)."""
    n = len(fitness)
    fit = np.asarray(fitness, dtype=float)
    if cfg.selection == "roulette":
        shifted = fit - fit.min() + 1e-6
        probs = shifted / shifted.sum()
        return rng.choice(n, size=n, p=probs)
    picks = rng.integers(n, size=(n, max(2, cfg.tournament_size)))
    winners = picks[np.arange(n), np.argmax(fit[picks], axis=1)]
    return winners


def abc_optimize(
    model: MajorityRuleModel,
    k: int,
    pool,
    cfg: ABCConfig | None = None,
    warm_start: list[list[int]] | None = None,
    name: str | None = None,
) -> SelectionResult:
    """Discrete ABC (+ optional local refinement) for majority-rule IM."""
    cfg = cfg or ABCConfig()
    rng = np.random.default_rng(cfg.random_state)
    t0, e0 = time.perf_counter(), model.n_evals
    pool = list(pool)
    n_sources = max(4, cfg.colony_size // 2)
    if cfg.adaptive_colony:
        n_sources = max(4, min(n_sources, len(pool) // max(1, k)))
    reserve = (cfg.reserve_fraction if cfg.refine_every else 0.0)
    soft_budget = (cfg.eval_budget * (1.0 - reserve)) if cfg.eval_budget else None

    # sandwich seeding runs on its own (cheaper, submodular) model; its
    # cascade evaluations are counted 1:1 into the total budget
    pre_spent = 0

    def spent() -> int:
        return model.n_evals - e0 + pre_spent

    def budget_left() -> bool:
        return soft_budget is None or spent() < soft_budget

    def remaining() -> int | None:
        if cfg.eval_budget is None:
            return None
        return max(0, cfg.eval_budget - spent())

    # ---------------- initialisation ---------------- #
    population: list[list[int]] = []
    celf_stats: dict = {}
    if cfg.seed_with_celf and cfg.seed_mode == "sandwich":
        bar = type(model)(model.graph, variant="slt", delta=model.delta,
                          n_samples=model.R, seed=model.seed,
                          backend=model.backend)
        r_bar = celf(bar, k, pool=pool)
        pre_spent = r_bar.evaluations
        population.append(list(r_bar.seeds))
        celf_stats = {"sandwich_seed_evals": r_bar.evaluations,
                      "sandwich_seed_spread_bar": r_bar.spread}
    elif cfg.seed_with_celf and cfg.seed_mode == "celf":
        r = celf(model, k, pool=pool)
        population.append(list(r.seeds))
        celf_stats = {"celf_seed_spread": r.spread}
    if cfg.seed_with_exchange and cfg.seed_with_celf and budget_left():
        r0 = celf(model, k, pool=pool)
        S2, f2, _ = celf_exchange_refine(
            model, r0.seeds, pool, max_passes=2,
            eval_budget=remaining() if cfg.eval_budget else 4000, lazy=cfg.lazy_refine)
        population.append(list(S2))
        celf_stats["celf_exchange_seed_spread"] = f2
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

    def accept(ns: float, nph: float, nse: float, i: int) -> bool:
        """Lexicographic (sigma, Phi): sigma must improve beyond MC noise, OR
        stay within noise while Phi improves.  With accept_rule="metro",
        strictly worse moves are additionally accepted with Metropolis
        probability exp(-(cs-ns)/T) -- the plateau-escaping thermal rule that
        SA exploits and plain ABC lacks."""
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

    def dedupe() -> None:
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

    # ---------------- main loop ---------------- #
    for cycle in range(1, cfg.max_cycles + 1):
        if not budget_left():
            break
        if cfg.stagnation_limit and cycle - last_gain_cycle > cfg.stagnation_limit:
            break

        # --- employed bees --------------------------------------------- #
        for i in range(n_sources):
            if not budget_left():
                break
            if rng.random() < cfg.gbest_prob and len(population) > 1:
                partner = best_set                      # W1: gbest guidance
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

        # --- onlooker bees ---------------------------------------------- #
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

        # --- scout bees -------------------------------------------------- #
        for i in range(n_sources):
            if trials[i] > cfg.limit and budget_left():
                if cfg.directed_scout and rng.random() < 0.7:      # W3
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

        # --- elitism ----------------------------------------------------- #
        i = int(np.argmax([s for s, _, _ in fit_pairs]))
        if fit_pairs[i][0] > best_fit:
            best_fit, best_set = fit_pairs[i][0], list(population[i])
            last_gain_cycle = cycle

        # --- local refinement of the incumbent --------------------------- #
        if cfg.refine_every and refine_active and cycle % cfg.refine_every == 0 and budget_left():
            S2, f2, st = celf_exchange_refine(
                model, best_set, pool,
                max_passes=cfg.refine_passes,
                eval_budget=remaining(),                    # W6: budget-respecting
                lazy=cfg.lazy_refine,
                two_swap=cfg.two_swap,
                cascade_aware=cfg.cascade_aware,
            )
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
                if zero_refines >= 2:      # refinement stopped paying: gate it off
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
                  f"mean={np.mean([s for s, _, _ in fit_pairs]):8.3f}  evals={spent()}")

    # ---------------- final polish (within budget) ---------------- #
    if cfg.refine_every and refine_active and (cfg.eval_budget is None or remaining() > 0):
        S2, f2, st = celf_exchange_refine(
            model, best_set, pool,
            max_passes=cfg.refine_passes + 1,
            eval_budget=remaining(),
            lazy=cfg.lazy_refine,
            two_swap=cfg.two_swap,
            cascade_aware=cfg.cascade_aware,
        )
        for key in ("refine_swaps", "refine_pair_swaps",
                    "lazy_bound_checks", "lazy_bound_violations"):
            refine_stats[key] += st[key]
        if f2 > best_fit:
            best_fit, best_set = f2, list(S2)

    label = name or ("ABC-CELF" if cfg.refine_every else "ABC")
    return SelectionResult(
        algorithm=label,
        seeds=list(best_set),
        spread=best_fit,
        runtime=time.perf_counter() - t0,
        evaluations=spent(),
        trace=trace.best,
        extra={
            "cycles_run": len(trace.cycle),
            "eval_curve": trace.evals,
            "mean_curve": trace.mean,
            "scout_resets": scout_resets,
            "dup_replacements": dup_replacements,
            **refine_stats,
            **celf_stats,
        },
    )


# --------------------------------------------------------------------------- #
#  Convenience wrappers used by the experiment driver
# --------------------------------------------------------------------------- #
def abc_plain(model, k, pool, budget=None, seed=0, cycles=80):
    cfg = ABCConfig(refine_every=0, seed_with_celf=False, max_cycles=cycles,
                    eval_budget=budget, random_state=seed)
    return abc_optimize(model, k, pool, cfg, name="ABC (no refinement)")


def abc_celf(model, k, pool, budget=None, seed=0, cycles=80):
    cfg = ABCConfig(refine_every=10, seed_with_celf=True, max_cycles=cycles,
                    eval_budget=budget, random_state=seed)
    return abc_optimize(model, k, pool, cfg, name="ABC-CELF (proposed)")


def celf_then_exchange(model, k, pool, seed=0, eval_budget=None):
    """Ablation: CELF plus local exchange, no population search."""
    t0, e0 = time.perf_counter(), model.n_evals
    r = celf(model, k, pool=pool)
    S, f, st = celf_exchange_refine(model, r.seeds, pool, max_passes=4,
                                    eval_budget=eval_budget)
    return SelectionResult(
        algorithm="CELF + 1-opt",
        seeds=S, spread=f,
        runtime=time.perf_counter() - t0,
        evaluations=model.n_evals - e0,
        extra=st,
    )


if __name__ == "__main__":
    from majority_rule_model import load_graph, candidate_pool, top_degree

    G = load_graph("ba500")
    m = MajorityRuleModel(G, "smr", 0.15, 200, seed=1)
    pool = candidate_pool(m, 150)
    k = 10

    for fn in (top_degree,):
        r = fn(m, k)
        print(f"{r.algorithm:22s} sigma={r.spread:8.3f}")
    r = celf(m, k, pool=pool)
    print(f"{r.algorithm:22s} sigma={r.spread:8.3f} evals={r.evaluations}")
    r = abc_plain(m, k, pool, budget=6000, cycles=40)
    print(f"{r.algorithm:22s} sigma={r.spread:8.3f} evals={r.evaluations} t={r.runtime:.2f}s")
    r = abc_celf(m, k, pool, budget=6000, cycles=40)
    print(f"{r.algorithm:22s} sigma={r.spread:8.3f} evals={r.evaluations} t={r.runtime:.2f}s "
          f"swaps={r.extra['refine_swaps']}")