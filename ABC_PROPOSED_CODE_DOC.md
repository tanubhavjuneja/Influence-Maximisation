# ABC (proposed) — Complete Code Documentation

**File documented:** `abc_celf.py` (688 lines)
**Author of the solution:** the project's optimised Artificial Bee Colony,
used as `ABC (proposed)` in every experiment in this repository.

This document walks through the *entire file* — every class, function,
parameter and design decision — so the code can be audited, reproduced and
extended without re-reading the source.

---

## 0. What the file is

`abc_celf.py` contains:

1. `celf_exchange_refine(...)` — a lazy 1-opt local-search refinement with
   cascade-aware candidate restriction and a 2-swap shaking phase
   (the "CELF-style local exchange" half of the hybrid).
2. `ABCConfig`, `ABCTrace` — configuration and tracing dataclasses.
3. `abc_optimize(...)` — the discrete ABC metaheuristic (the "ABC" half).
4. Convenience wrappers — `abc_plain`, `abc_celf`, `celf_then_exchange` —
   used by the experiment drivers and ablations.

The design principle stated at the top of the file:

> Forward greedy (and CELF) builds a seed set one node at a time and is
> driven by *singleton marginal gains*. Under majority rule those gains are
> nearly all zero, so greedy is choosing among ties and is effectively blind.
> Influence under majority rule is a property of **coalitions**, not of
> individuals. ABC searches the space of whole size-k sets; the local
> refinement supplies the node-level gradient ABC lacks.

The file imports only `heapq`, `time`, `numpy`, and from
`majority_rule_model`: `MajorityRuleModel`, `SelectionResult`, `celf`.
`SelectionResult` is the universal result object consumed by every driver:

```python
SelectionResult(algorithm, seeds, spread, runtime, evaluations,
                trace, extra={...})
```

---

## 1. `celf_exchange_refine` — the local refinement (lines 113–265)

```python
def celf_exchange_refine(
    model, seeds, pool,
    max_passes=3, eval_budget=None,
    lazy=True, audit=True,
    cascade_aware=True, max_cands=60,
    two_swap=True, two_swap_tries=400,
) -> tuple[list[int], float, dict]:
```

**Purpose.** Take a seed set and improve it by 1-opt exchanges (replace one
seed with one pool node), then by random 2-swaps. All evaluations go through
`model.sigma(...)`, so the refinement works on any diffusion model exposing
the `MajorityRuleModel` interface (unsigned majority, signed majority, IC).

### 1.1 Lazy 1-opt loop (lines 173–230)

* Iterates positions `pos` of the current set `S`; for each, builds
  `S_minus = S \ {S[pos]}` and evaluates `base = sigma(S_minus)`.
* **Candidates** (`cands`, line 185): all pool nodes not in `S_minus`. If
  `cascade_aware` and the pool is bigger than `max_cands` (60), the
  candidates are restricted by `_cascade_cands` to cascade-relevant nodes.
* **Lazy mode** (`lazy=True`, lines 191–216): each candidate `v` carries a
  heap bound equal to its last marginal gain against a *subset* of the
  current base. Initially the bound is the classic CELF singleton bound
  `sigma({v}) - sigma({})` (pulled from the model's evaluation cache when
  available, lines 195–200). The heap pops the most promising candidate;
  if its bound was computed against the *current* base (key matches
  `base_key`), it is accepted without re-evaluation; otherwise it is
  re-evaluated against the current base (`gain = sigma(S_minus+[v]) - base`,
  line 211) and pushed back with the fresh bound.
* **Audit** (lines 212–215): because majority-rule spread is **not**
  submodular, a stale bound is not a valid upper bound. Every re-evaluation
  whose fresh gain exceeds its stale bound is counted in
  `lazy_bound_violations`; `lazy_bound_checks` counts audited
  re-evaluations. This is the honest bookkeeping that turns "CELF on a
  non-submodular function" from a silently wrong heuristic into a
  *documented, audited* heuristic.
* **Exact mode** (`lazy=False`, lines 217–222): every candidate is evaluated
  against the current base; no bound is trusted. Used in ablations.
* **Budget**: `over()` (line 170) compares fresh evaluations spent
  (`model.n_evals - start_evals`) against `eval_budget`; cache hits are free.
* **Acceptance** (line 224): the best candidate is accepted only if it
  strictly improves `best` and differs from the removed node. `max_passes`
  full passes are attempted; a pass with no improvement terminates the loop
  (line 229).

### 1.2 2-swap shaking (lines 232–256)

Repairs coalitions 1-opt cannot move (weakness W9). It performs up to
`two_swap_tries` (400) random 2-for-2 exchanges — remove two seeds, insert
two pool nodes — accepting the first strict improvement, for at most 2 outer
rounds. Counted in `refine_pair_swaps`. The RNG is seeded fixed (`default_rng(7)`)
for reproducibility.

### 1.3 Returned stats (lines 258–265)

```python
{"refine_swaps": ..., "refine_pair_swaps": ..., "refine_evals": ...,
 "lazy_bound_checks": ..., "lazy_bound_violations": ...}
```

These flow into the result CSV as the `x_refine_*` and `x_lazy_*` columns.

### 1.4 `_cascade_cands` (lines 113–127)

Restricts swap-in candidates to **neighbours of the current seeds** and of
**near-threshold nodes** (`model._border_nodes(S_minus, 30)`: nodes whose
activation slack ≤ 1 at convergence). If fewer than 4 candidates survive, it
falls back to the full pool. This encodes weakness W9's insight: only
cascade-relevant insertions can change the outcome, so the neighbourhood is
shrunk without losing useful moves.

---

## 2. `ABCConfig` (lines 271–304)

Every tunable knob of the metaheuristic, one dataclass. Defaults in brackets.

| Field | Default | Role |
|---|---:|---|
| `colony_size` | 30 | population size; food sources = `colony_size // 2` (min 4) |
| `max_cycles` | 80 | main-loop cycles |
| `limit` | 6 | abandonment threshold that triggers a scout |
| `mutation_rate` | 0.2 | prob. of an extra random swap per candidate (W4) |
| `n_mod` | 1 | number of partner-guided swaps per candidate (W4) |
| `gbest_prob` | 0.6 | prob. the partner is the global best (W1) |
| `selection` | "tournament" | onlooker source choice: `tournament` or `roulette` (W2) |
| `tournament_size` | 4 | tournament bracket size |
| `directed_scout` | True | rebuild abandoned sources near the incumbent (W3) |
| `dedupe` | True | no duplicate food sources (W5) |
| `accept_equal` | True | slide along exact plateaus (W7 safety net) |
| `explore_prob` | 0.15 | prob. a bee move is a fresh random set (W7) |
| `potential_weight` | 0.05 | λ in fitness = σ + λ·Φ (W7 plateau gradient) |
| `tol_mult` | 2.0 | σ-ties are differences within `tol_mult × SEM` |
| `accept_rule` | "lex" | `"lex"` = lexicographic (σ, Φ); `"metro"` = Metropolis |
| `temp_frac` | 0.01 | Metropolis initial temperature (fraction of best σ) |
| `temp_min` | 0.25 | temperature floor |
| `temp_decay` | 0.97 | per-cycle temperature decay |
| `refine_every` | 10 | local-refinement cadence (0 = never) |
| `refine_passes` | 2 | 1-opt passes per refinement call |
| `lazy_refine` | True | lazy (audited) 1-opt vs exact |
| `two_swap` | True | 2-swap shaking when 1-opt stalls (W9) |
| `cascade_aware` | True | restrict swap-ins to cascade-relevant nodes (W9) |
| `seed_mode` | "celf" | warm start: `"celf"`, `"sandwich"`, or none |
| `seed_with_celf` | True | seed the population with CELF's output |
| `seed_with_exchange` | False | also seed with CELF+exchange |
| `adaptive_colony` | False | shrink colony to pool/k when the pool is tiny |
| `reserve_fraction` | 0.15 | share of the budget reserved for refinement (W6) |
| `stagnation_limit` | 0 | early stop after this many gainless cycles (0 = off) |
| `eval_budget` | None | hard cap on fresh cascade evaluations |
| `random_state` | 0 | RNG seed |
| `verbose` | False | per-cycle console trace |

**Config actually used in the real-network runs** (from
`run_real_networks.py`, `solve_majority` / `solve_ic`):

```python
ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
          seed_with_celf=True, eval_budget=budget,
          random_state=600 + rs)
```

i.e. budget = k × 200, refinement disabled (refine_every=0) so all budget
goes to the colony, CELF warm start on, 3 restarts (Bitcoin) / 2 (Slashdot).

## 3. `ABCTrace` (lines 307–312)

Holds per-cycle records: `cycle` (cycle numbers), `best` (best σ),
`mean` (mean σ of the population), `evals` (cumulative evaluations).
`SelectionResult.trace` receives `trace.best`; `x_cycles_run`,
`x_eval_curve`, `x_mean_curve` in the CSV come from here.

---

## 4. Operators

### 4.1 `_biased_set` (lines 315–320)

Population initialisation. Draws k nodes from the pool **without
replacement, proportional to a weight vector**:

* unsigned graphs: `model.deg` (degree-biased);
* signed graphs: `model.pool_weight` (net positive support — sign-aware
  initialisation for the signed model).

`_degree_biased_set` is a backwards-compatible alias (line 323).

### 4.2 `_neighbour_solution` (lines 326–355)

The discrete analogue of Karaboga's `v_i = x_i + φ (x_i − x_j)`.
The "difference" of two k-sets is their symmetric difference: `n_mod` times,
drop a node unique to `x` and adopt a node unique to the `partner`
(lines 337–348); if the sets are identical, a pool node is swapped in at
random. With probability `mutation_rate` an extra purely random replacement
is applied (W4). Duplicates are removed and the result truncated to k.

### 4.3 `_candidate` (lines 358–362)

One bee move: with probability `explore_prob` a fresh uniform-random k-set
(W7 diversity injection); otherwise `_neighbour_solution`.

### 4.4 `_pick_sources` (lines 365–375)

Onlooker source selection. `"roulette"`: fitness-proportional probabilities
(shifted to be non-negative). `"tournament"` (default): for each of the n
onlookers, sample `tournament_size` sources uniformly and pick the fittest
(W2 — immune to premature roulette convergence when fitnesses are
near-equal).

---

## 5. `abc_optimize` — the main solver (lines 378–638)

```python
def abc_optimize(model, k, pool, cfg=None, warm_start=None,
                 name=None) -> SelectionResult:
```

### 5.1 Setup (lines 387–410)

* `n_sources = max(4, colony_size // 2)` — food sources (= employed bees);
  optionally shrunk by `adaptive_colony`.
* Budget bookkeeping: `spent()` counts fresh evaluations since entry plus
  `pre_spent` (sandwich-seeding evaluations, counted 1:1); `remaining()`
  is the budget left for refinement. `soft_budget` reserves
  `reserve_fraction` of the budget for refinement (W6).

### 5.2 Initialisation (lines 412–441)

1. **CELF warm start** (W guarantee 1): `seed_mode="celf"` runs
   `celf(model, k, pool)` and puts its seed set first in the population;
   `seed_mode="sandwich"` runs CELF on the *sandwich* model (see §7) and
   records the submodular upper bound statistics.
2. Optional CELF+exchange seed (`seed_with_exchange`).
3. Optional external `warm_start` sets.
4. Remaining sources filled by `_biased_set`, truncated to `n_sources`.

### 5.3 Fitness (lines 443–448)

```python
def fitness_of(s):
    if cfg.potential_weight > 0:
        sg, ph, se = model.sigma_potential(s)
        return sg, sg + cfg.potential_weight * ph, se
    return (sigma, sigma, 0.0)
```

`s`-level sigma, Phi (plateau potential), and the standard error `se` of the
Monte-Carlo estimate. Phi is computed **in the same cascade pass** as sigma
(`sigma_potential`), so the plateau gradient (W7) costs nothing extra.
`fitness` (the scalar used for selection) = σ + λ·Φ.

### 5.4 Acceptance rule (lines 467–482)

Lexicographic (σ, Φ) with a noise-aware tolerance:

1. **Strict improvement:** accept if `ns > cs + tol` with
   `tol = tol_mult × max(se_current, se_new)` — the move must beat the
   Monte-Carlo noise floor.
2. **Within noise:** if `|ns − cs| ≤ tol`, accept if Φ improves (or ties,
   when `accept_equal`) — this lets bees slide across zero-σ plateaus.
3. **Metropolis option** (`accept_rule="metro"`): strictly worse moves are
   additionally accepted with probability `exp((ns−cs)/T)` — the
   plateau-escaping thermal rule SA exploits and plain ABC lacks. `temp` is
   initialised to `temp_frac × best σ` and decays geometrically (lines 465,
   598–600).

### 5.5 The main cycle loop (lines 498–603)

Each cycle runs six phases in order:

1. **Employed bees** (505–522): each source generates a candidate — with
   probability `gbest_prob` the partner is the global best (W1), otherwise a
   random other source — and the candidate replaces the source iff
   `accept(...)` passes; otherwise `trials[i] += 1`.
2. **Onlooker bees** (524–543): sources are chosen by `_pick_sources`
   (tournament by default) and the same accept/replace step is applied.
3. **Scout bees** (545–558): a source with `trials[i] > limit` is abandoned.
   With probability 0.7 the **directed scout** (W3) rebuilds it by a strong
   perturbation of the incumbent
   (`_neighbour_solution(best, best, n_mod=2, mutation_rate=1.0)`);
   otherwise a uniform random set. `scout_resets` counts resets.
4. **Duplicate suppression** (560–561, 484–495): `dedupe()` replaces any
   duplicate set (same `frozenset`) with a fresh `_biased_set` (W5);
   counted in `dup_replacements`.
5. **Elitism** (563–567): the best source updates the incumbent if it
   strictly beats it; `last_gain_cycle` records the gain cycle.
6. **Local refinement** (569–592): every `refine_every` cycles the incumbent
   is polished with `celf_exchange_refine` under `remaining()` budget (W6).
   On improvement the incumbent is replaced and the worst source is
   overwritten by the refined set. If two consecutive refinements make zero
   swaps, `refine_active` is switched off (W8 — the refinement stopped
   paying, its budget goes back to the colony).

`stagnation_limit` (line 501) breaks the loop after too many gainless
cycles. The trace records (σ best, σ mean, evals) every cycle.

### 5.6 Final polish (lines 605–619)

One last refinement pass with `refine_passes + 1` on the incumbent, inside
the remaining budget.

### 5.7 Result (lines 621–638)

Returns a `SelectionResult` named `name` (default `"ABC-CELF"` if
`refine_every` else `"ABC"`) with seeds, best σ, runtime, evaluation count,
trace, and `extra` carrying:

| Key | Meaning |
|---|---|
| `cycles_run` | cycles actually executed |
| `eval_curve`, `mean_curve` | per-cycle evaluation / mean-σ curves |
| `scout_resets` | scout resets performed |
| `dup_replacements` | duplicate sources replaced |
| `refine_swaps` / `refine_pair_swaps` | 1-opt / 2-swap improvements |
| `lazy_bound_checks` / `lazy_bound_violations` | lazy-refinement audit |
| `celf_seed_spread` | spread of the CELF warm start |

(These become the `x_*` columns of `results/real_*.csv`.)

---

## 6. Wrappers (lines 641–668)

* `abc_plain(model, k, pool, budget, seed, cycles)` — pure ABC, no CELF seed,
  no refinement (`refine_every=0, seed_with_celf=False`); ablation name
  "ABC (no refinement)".
* `abc_celf(model, k, pool, budget, seed, cycles)` — full proposed solver
  (`refine_every=10, seed_with_celf=True`), name "ABC-CELF (proposed)".
* `celf_then_exchange(model, k, pool, ...)` — ablation: CELF + 4-pass local
  exchange, no population search; name "CELF + 1-opt".

The `__main__` block (lines 671–688) is a self-test on the synthetic
`ba500` graph comparing Degree, CELF, plain ABC and ABC-CELF.

---

## 7. Sandwich seeding (opt-in guarantee machinery)

When `seed_mode="sandwich"` (lines 415–423), CELF is run on the concave
**envelope** model: `variant="slt"` (LT-type, θ ~ U[0, 0.5+δ]) with
F̄(x) = min(x/(0.5+δ), 1). For this model σ_bar ≥ σ pathwise and σ_bar is
submodular (Mossel–Roch sufficiency), so CELF is *valid* there and yields:

* a submodular-valid warm start,
* an upper bound `OPT ≤ σ_bar(S_celf_bar) · e/(e−1)`,
* an after-the-fact guarantee
  `σ(S) ≥ (1−1/e) · OPT / α(S)` with `α(S) = σ_bar(S)/σ(S)`.

Its evaluations are counted 1:1 into the budget (`pre_spent`). The real
runs use `seed_mode="celf"` (the cheaper warm start); the sandwich mode is
the guaranteed variant used in the synthetic study.

---

## 8. Where the literature upgrades live (W1–W9)

| Weakness | Fix | Code |
|---|---|---|
| W1 weak exploitation | gbest-guided partner | lines 508–509, 529–530 (`gbest_prob`) |
| W2 roulette convergence | tournament onlookers | `_pick_sources` 365–375 |
| W3 blind scouts | directed scouts | lines 548–551 |
| W4 fixed perturbation | `n_mod` + `mutation_rate` | `_neighbour_solution` 326–355 |
| W5 duplicates | `dedupe()` | 484–495 |
| W6 refinement budget | reserve + `remaining()` | 394–395, 571–577 |
| W7 plateaus | σ+λ·Φ gradient, `explore_prob` | 443–448, 358–362 |
| W8 stale refinement | adaptive gating off | 589–592 |
| W9 broken coalitions | cascade-aware cands + 2-swap | 113–127, 232–256 |

---

## 9. How the solver is invoked in the real-network pipeline

`run_real_networks.py` lines ~335–339 (`solve_majority`) and ~383–387
(`solve_ic`):

```python
cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                seed_with_celf=True, eval_budget=budget,
                random_state=600 + rs)
return abc_optimize(model, k, pool, cfg, name="ABC (proposed)")
```

* `budget = k × |pool|` with `|pool| = 200` — the same budget Greedy costs.
* 3 restarts on the Bitcoin graphs, 2 on Slashdot (`random_state=600+rs`).
* The same call works for the signed majority model (`SignedMajorityRuleModel`
  exposes the same interface including `pool_weight`).

---

## 10. One-line summary of each piece

| Line | Symbol | One line |
|---|---|---|
| 113 | `_cascade_cands` | restrict swap-ins to cascade-relevant nodes |
| 130 | `celf_exchange_refine` | lazy audited 1-opt + 2-swap shaking |
| 271 | `ABCConfig` | all solver knobs in one dataclass |
| 307 | `ABCTrace` | per-cycle best/mean/eval records |
| 315 | `_biased_set` | degree / net-positive-support-biased init |
| 326 | `_neighbour_solution` | symmetric-difference ABC move |
| 358 | `_candidate` | move with exploration probability |
| 365 | `_pick_sources` | tournament / roulette onlooker choice |
| 378 | `abc_optimize` | the full proposed solver |
| 644 | `abc_plain` | ablation: pure ABC |
| 650 | `abc_celf` | the full hybrid (name "ABC-CELF") |
| 656 | `celf_then_exchange` | ablation: CELF + exchange only |