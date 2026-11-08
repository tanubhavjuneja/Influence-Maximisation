# Influence Maximisation under the Majority Rule Diffusion Model

## Monotone but not submodular: structural analysis, a literature-driven Artificial Bee Colony optimiser, and an honest equal-budget, held-out evaluation (including a signed-graph extension)

**Technical Analysis Report — Final Version**

**Scope.** (1) A machine-checked proof that the majority-rule diffusion model is monotone but **not** submodular, with the Mossel–Roch sufficient condition as the structural explanation; (2) the measurable consequences for greedy-style seed selection — in particular a decision-relevant audit showing CELF's lazy bound failing on a majority of its pruning decisions; (3) a discrete Artificial Bee Colony optimiser whose operators are chosen from the ABC-improvement literature and selected by a leave-one-out ablation; (4) an evaluation protocol with **equal evaluation budgets**, **held-out scoring on fresh threshold draws (R = 1000)**, and the **graph as the statistical unit** across 16–8 fresh instances per graph family; (5) a signed/weighted extension of the model with a non-monotonicity witness and a sign-ablation control; (6) a controlled δ-sweep from the deterministic majority rule to the (submodular) Linear Threshold model, with the confound explicitly diagnosed.

All numbers below are reproducible from the scripts and CSVs in this directory (`run_experiments.py`, `run_validation.py`, `extend_ws.py`, `theory_checks.py`, `results/`). Every long-running study appends its results after each cell and resumes after interruption.

---

## Executive Summary

The brief asked for a proof that the majority-rule diffusion model is submodular and monotone. **Monotonicity is true and provable; submodularity is false**, for any genuine majority rule, and the false half is the half that matters. A strict-majority threshold makes a node's local activation function convex rather than concave on its lower stretch — precisely where the Mossel & Roch (2007) sufficient condition for submodularity in the general threshold model fails — and the failure is machine-checked: 23–76% of non-degenerate (S ⊆ T, v) triples violate diminishing returns on every majority variant tested, against a clean 0.0% for the submodular Linear Threshold control.

This is not a technicality. **CELF** — the textbook lazy-greedy accelerator — degrades measurably: its lazy bound is violated on **50.2%** of re-checks across all runs, its chosen set differs from exact Greedy's in **16 of 17** (graph, k) configurations and scores **strictly below Greedy in 13 of 17**; on hub-heavy graphs it collapses to ~25% of the spread achieved by a plain degree ranking (ba500, k = 30: CELF 131.9 vs Degree 473.7 vs ABC 493.8, held-out).

The optimiser proposed here is a discrete Artificial Bee Colony (ABC) whose operators address the documented weaknesses of the canonical algorithm (Karaboga 2005 TR06; Karaboga & Basturk 2007): gbest-guided moves (Zhu & Kwong 2010), tournament onlookers, directed scouts (Kiran & Fındık 2015), duplicate suppression, a plateau-gradient tie-break Φ(S) = Σ_v min(1, net_v/need_v), and a cascade-aware 1-opt + 2-swap local search. The configuration was fixed by a leave-one-out ablation on 13 tuning graphs disjoint from the test graphs (26 configs × 20 seeds): the gbest guide is the only operator whose removal significantly hurts; the CELF-style refinement *reservation* significantly hurts and was removed — the method is therefore presented as **ABC**, not "ABC-CELF".

**Headline result (pre-specified primary comparison).** On **16 independently generated Watts–Strogatz instances** (degree ∈ {6, 8, 10}, seeds varying), with each graph averaged over k ∈ {5, 10, 20} and all solvers sharing the same evaluation budget: **ABC beats Greedy by +4.57 held-out spread, winning 14 of 16 graphs (bootstrap CI [+2.46, +6.91], Wilcoxon p = 3.1 × 10⁻⁴** — below the Bonferroni threshold 0.0056 for the 9 family tests). The advantage is concentrated **below Greedy's cascade threshold k\*** (the smallest k at which Greedy's seed set triggers any cascade): +65% relative gain below k\* vs +23% at/above, and on the random-percolation control, *no* random k-set cascades for k ≤ 20 — majority cascades require **chosen coalitions**, which is exactly what set-level search provides.

**Equally important, stated plainly:** ABC is not the best set-level searcher. **Simulated annealing (SA) with the same swap neighbourhood is at least as good**: on WS, SA − ABC = +1.41 (CI [−2.84, −0.12], SA favoured); on Barabási–Albert, +0.49 (p = 0.039). Both beat Greedy on WS (SA by +5.99, 15/16); both tie Greedy on hub-heavy BA graphs; both beat CELF everywhere. ABC's contributions are its budget efficiency (it reaches its ceiling at 1× budget where SA needs 4×), its ≥CELF guarantee, and its run-to-run stability (CV 1.7%).

**Signed extension.** The model generalises to signed, weighted edges via the net-support rule; it is non-monotone (constructive hub-blocker example; a drop-0.26 witness located on a signed BA graph). A sign-ablation control (identical topology, all-positive twin) shows the ABC-vs-Greedy advantage persists — even grows — without the signs (+181% vs +95% on sbm120, k = 15): **the search advantage is the plateau effect, not the sign structure.** The signed contribution is the model generalisation and the non-monotonicity witness.

**The δ-sweep claim from earlier drafts is withdrawn.** Correlating the ABC advantage with δ or with γ across 8 δ-values on one graph is confounded (both move with δ); a partial correlation controlling for δ removes every association (p ≥ 0.31), and a fixed-δ test across the 24 multi-instances shows no reliable correlation either (ρ = −0.23, p = 0.28). The structural story stands on the proofs, the counterexamples and the CELF audit — not on the sweep.

**Recommendation.** Present: (i) prove monotone; (ii) disprove submodularity and identify why (Mossel–Roch sufficiency + exhaustive scans); (iii) quantify γ as the honest replacement for (1−1/e); (iv) show CELF's failure with the audit; (v) show that set-level search beats marginal search on sparse clustered graphs below the contagion threshold, and that ABC is a valid, budget-efficient, guarantee-carrying member of that class — while SA is at least as good.

---

## 1. Problem Statement and Its Correction

### 1.1 The diffusion model

Progressive (once active, always active) threshold diffusion on an undirected graph G = (V, E). Each node v independently draws a susceptibility threshold θ_v once, and keeps it:

|{active neighbours of v}|  >  θ_v · deg(v).        (1)

σ(S) = E_θ[|A∞(S, θ)|]. Threshold vectors are sampled once per model instance and reused for every evaluation (common random numbers), which removes sampling noise from marginal-gain comparisons; each cascade is simulated as batched sparse matrix–matrix products (GPU-accelerated on the RTX 3060; CPU fallback automatic).

Variants: **dmr** (θ = ½ a.s.), **smr** (θ ~ U[½−δ, ½+δ], workhorse δ = 0.15), **lt** (θ ~ U[0,1] — submodular control), **slt** (θ ~ U[0, ½+δ] — the sandwich upper-bound model, §2.6). Signed extension (§8): edge sign/weight attributes; node v activates iff (positive active weight − negative active weight) > θ_v · deg(v); for all-positive graphs this reduces exactly to (1).

### 1.2 What the brief asked, and why half of it cannot be proved

"Prove submodular and monotone, then optimise." Monotonicity is true (§2.1). Submodularity is false for every genuine majority rule (§2.2–2.4), for a structural reason (non-concavity of the threshold CDF, §2.3). The deliverable therefore becomes: prove monotonicity; disprove submodularity and identify why; quantify the instance-specific submodularity ratio γ as the honest replacement for the textbook (1−1/e) guarantee; and optimise with a method whose design is motivated by — not merely applied despite — that finding.

---

## 2. Structural Analysis: Monotone, Not Submodular

### 2.1 Proposition 1 — Monotonicity (proved)

**Claim.** σ is monotone non-decreasing: S ⊆ T ⟹ σ(S) ≤ σ(T).

**Proof.** Fix a threshold realisation θ and run the two cascades from S and T in lock-step. Induct on t with A_t(S) ⊆ A_t(T). Base: A₀(S) = S ⊆ T = A₀(T). Step: if v activates at t+1 under S, then |N(v) ∩ A_t(S)| > θ_v deg(v); by induction A_t(S) ⊆ A_t(T), so the same inequality holds under T and v activates there too. Progressivity preserves the invariant. Hence |A∞(S,θ)| ≤ |A∞(T,θ)| pointwise, and taking expectations preserves the inequality. ∎

**Verification.** 5,800 sampled pairs (600 + 600 + 250 per configuration × 12 configurations: 3 graphs × 4 variants) — **0 violations** (Table 1).

### 2.2 Proposition 2 — Non-submodularity: the minimal counterexample

Star K₁,₃: centre w adjacent to leaves x, y, z (deg(w) = 3 ⟹ w needs 2 active neighbours). Under dmr:

| Coalition | Gain of adding y |
|---|---|
| S = ∅ | σ({y}) − σ(∅) = **1.0** (y cannot flip w alone) |
| T = {x} | σ({x,y}) − σ({x}) = **3.0** (x and y flip w; w then flips z: 4 active vs 1) |

S ⊆ T but the marginal gain is larger for the bigger set — increasing returns, violating submodularity. y is a *swing voter*: worthless alone, decisive inside the winning coalition.

**Does softening rescue submodularity?** Re-running the counterexample with θ ~ U[½−δ, ½+δ] (R = 20,000):

| δ | variant | gain(S,y) | gain(T,y) | submodular? |
|---|---|---|---|---|
| 0.00 | dmr | 1.00 | 3.00 | no (+2.00) |
| 0.05 | smr | 1.00 | 3.00 | no |
| 0.10 | smr | 1.00 | 3.00 | no |
| 0.20 | smr | 1.25 | 2.58 | no (+1.33) |
| 0.30 | smr | 1.65 | 1.90 | no (+0.25) |
| 0.50 | lt | 1.98 | 1.35 | **yes** (control) |

The K₁,₃ violation is clear for δ ≲ 0.3. **Larger stars keep it visible further**: K₁,₉ with T = 4 leaves shows the violation at δ = 0.35 (+0.40) and δ = 0.40 (+0.16), fading by δ = 0.45 (at δ = 0.49: −0.80, i.e. within noise of zero). The earlier draft's "every δ < 0.5" claim was wrong for K₁,₃ and is corrected here.

### 2.3 Why: the Mossel–Roch reduction to CDF concavity

Mossel & Roch (2007) proved that under the general threshold model, σ is submodular **if** every local activation function f_v(a) = F_θ(a/deg(v)) is submodular in the active-neighbour count a — a **sufficient** condition (they do not claim the converse). Since f_v depends on a single integer argument, submodularity of f_v is equivalent to **concavity of the threshold CDF** on the grid {0, 1/d, …, 1}.

Concavity says "each extra active neighbour helps less than the previous one". A majority rule says the opposite below the threshold: neighbours short of the majority contribute nothing, the one crossing the threshold is worth everything. θ ~ U[0,1] is the unique member of this family with a linear (concave *and* convex) CDF — which is why LT is submodular and every genuine majority rule fails the sufficient condition. Because the condition is only sufficient, the actual violations are then certified computationally (§2.2, §2.4).

### 2.4 Exhaustive computational verification

Every triple (S ⊆ T, v ∉ T) with |T| ≤ 3 was enumerated over a capped, degree-stratified node set, with a Monte-Carlo noise tolerance calibrated from σ's own sampling error (the LT control's clean 0.0% is the calibration check).

| graph | n / m | variant (δ) | monotone | non-deg. violation rate | γ_min |
|---|---|---|---|---|---|
| karate | 34 / 78 | lt (0.50) — control | ✓ | **0.0%** | 0.992 |
| karate | 34 / 78 | smr (0.30) | ✓ | 66.3% | 0.511 |
| karate | 34 / 78 | smr (0.15) | ✓ | 68.7% | 0.345 |
| karate | 34 / 78 | dmr | ✓ | 52.6% | 0.292 |
| lesmis | 77 / 254 | lt (0.50) — control | ✓ | **0.0%** | 0.967 |
| lesmis | 77 / 254 | smr (0.30) | ✓ | 64.4% | 0.462 |
| lesmis | 77 / 254 | smr (0.15) | ✓ | 60.4% | 0.653 |
| lesmis | 77 / 254 | dmr | ✓ | 47.0% | 0.333 |
| ba500 | 500 / 1491 | lt (0.50) — control | ✓ | **0.0%** | 0.963 |
| ba500 | 500 / 1491 | smr (0.30) | ✓ | 59.1% | 0.845 |
| ba500 | 500 / 1491 | smr (0.15) | ✓ | 75.7% | 0.693 |
| ba500 | 500 / 1491 | dmr | ✓ | 70.4% | 0.667 |

**Table 1.** Structural verification. Monotonicity holds everywhere (0 violations). Every majority variant shows large violation rates against a clean 0.0% LT control. γ_min across majority settings spans **0.292–0.845** (LT: 0.963–0.992). Note γ_min is a Monte-Carlo *minimum over samples*: it is an **optimistic** estimate of the true ratio, and the greedy guarantee (1 − e^(−γ)) should be read accordingly.

### 2.5 Hardness

For the deterministic threshold model with **general** thresholds, Lu, Zhang, Wu, Kim & Fu (J. Comb. Optim. 24(3):374–378, 2012) prove no polynomial-time n^(1−ε)-approximation unless P = NP. Our uniform-½ majority rule is a special case of their setting, so their inapproximability **does not automatically transfer**; approximation quality is therefore measured empirically against the brute-force optimum (§7.1) rather than claimed from hardness.

### 2.6 The sandwich upper bound (theory layer)

The concave envelope of the majority CDF on [0,1] is F̄(x) = min(x/(½+δ), 1), i.e. an LT-type model with θ ~ U[0, ½+δ] (`slt`). Since F̄ ≥ F pointwise, a common-uniform coupling gives **σ̄ ≥ σ pathwise**; F̄ concave ⟹ **σ̄ submodular** (Mossel–Roch sufficiency), so **CELF is valid on σ̄**. Consequences: an upper bound OPT ≤ σ̄(S)·e/(e−1), and an after-the-fact guarantee σ(S) ≥ (1−1/e)·OPT/α(S) with α(S) = σ̄(S)/σ(S). Measured α ≈ 13 on ba500 — the guarantee is **vacuous at small δ** (as expected), and the OPT bound is loose (≤ 1495 on ws1000 k = 15, true optimum ≈ 38). Sandwich-CELF is the weakest budgeted seeder (activation fraction 0.163, Table 7). It is kept as the theory layer, not as a competitive seeding.

---

## 3. Consequences for Greedy-Style Optimisation

### 3.1 The classical guarantee is void

Nemhauser–Wolsey–Fisher (1−1/e) requires submodularity; §2 shows it fails. The honest replacement is the Das & Kempe submodularity ratio γ (§2.4, Table 1), giving (1 − e^(−γ)) for greedy — an instance-specific, optimistic bound.

### 3.2 CELF specifically breaks — the decision-relevant audit

CELF's correctness rests on "stale marginal gain is an upper bound" — a submodularity consequence. The implementation audits every re-check:

- **Lazy-bound violation rate over all CELF runs: 50.24%** (418 violations / 832 re-checks). Replacing the earlier "89%" smoke-test figure, which came from 8 of 9 checks in a single run.
- **Decision audit** (`results/val_celf_audit.csv`): across 17 (graph, k) configs, CELF's chosen set differs from exact Greedy's in **16**, and scores **strictly below** Greedy in **13**; only 1 of the differing cases is a near-tie. Under submodularity CELF would reproduce Greedy exactly.

### 3.3 Where CELF collapses

On ba500, k = 30 (held-out): CELF 131.9 vs Greedy 491.5 vs Degree 473.7 vs ABC 493.8. CELF loses to a trivial degree ranking — a +273% gap that reflects CELF's collapse, not a generic metaheuristic miracle. Reporting convention throughout: lead with Greedy/Degree baselines and activation fractions (Table 7), and treat relative gains against CELF as "how far CELF falls", not "how good ABC is".

---

## 4. Proposed Method: Artificial Bee Colony with Local Search

### 4.1 Design rationale

Greedy/CELF are driven by singleton marginal gains, which are near-zero on a majority plateau: the information that matters — which coalitions jointly cross local majority thresholds — is invisible one node at a time. ABC searches whole size-k sets with swap operators, and can therefore *assemble and discard* coalitions (on signed graphs, even remove harmful seeds). Its weakness — no notion of local value — is supplied by a cascade-aware 1-opt + 2-swap local search. A Metropolis acceptance option exists (cfg `accept_rule="metro"`) but the default is the lexicographic (σ, Φ) rule; see §4.3.

### 4.2 Literature-driven upgrades over canonical ABC

Canonical ABC (Karaboga 2005 TR06; Karaboga & Basturk 2007) has well-documented weaknesses; each is addressed by an operator, following the ABC-improvement literature:

| # | canonical weakness | literature | this implementation |
|---|---|---|---|
| W1 | weak exploitation, slow late-stage convergence | gbest-guided ABC (Zhu & Kwong 2010) | partner = global best with prob. 0.6 |
| W2 | premature convergence of roulette wheel | tournament selection | tournament size 4 (roulette kept as option) |
| W3 | blind random scout restarts | directed scouts (Kiran & Fındık 2015) | rebuild abandoned sources near the incumbent |
| W4 | fixed perturbation strength | modification rate (Akay & Karaboga 2012) | n_mod partner swaps + mutation per candidate |
| W5 | duplicate food sources | — | duplicate suppression |
| W6 | refinement blowing the budget | — | remaining-budget-only + gating |
| W7 | rigid moves on plateaus ("rigid neighbourhood generation, weak directional guidance" — the ABC+VNS line of work on IM) | — | plateau gradient Φ as tie-break within MC noise; accept-equal; random exploration moves |
| W8 | 1-opt cannot repair broken coalitions | — | cascade-aware candidates (neighbours of seeds and of slack-≤1 nodes) + 2-swap shaking |
| W9 | (signed) no operator can remove harmful seeds | — | generic swap operators handle it; sign-aware initialisation via net positive support |

**Plateau gradient.** Φ(S) = Σ_v min(1, net_v/need_v) at cascade close — the smooth mass of "almost-activated" nodes σ cannot see — computed in the same cascade pass (zero extra evaluation cost). Acceptance is **lexicographic**: σ must improve beyond tol = 2 × MC standard error; within that band, Φ decides. Global σ + λΦ fitness was tested and *hurt* (ba500 k=10: 51.9 vs 57.5 held-out) — recorded as a negative result.

### 4.3 Configuration selection: leave-one-out ablation

Tuning on **13 graphs disjoint from the test graphs** (florentine, davis, star60, cycle60, grid100, tree63, barbell50, caveman60, windmill46, ba300, ws300, powerlaw300, sbm450), 26 (graph, k) configs × 20 seeds each, held-out scoring (`results/val_loo.csv`). Removing one operator at a time from the full configuration:

| variant | Δ mean held-out vs full | wins | p |
|---|---|---|---|
| −gbest | **−0.132** | 1/26 | **0.0006** |
| −tournament | +0.020 | 6/26 | 0.40 |
| −directed_scout | −0.013 | 6/26 | 0.68 |
| −dedupe | −0.006 | 5/26 | 0.25 |
| −accept_equal | −0.001 | 1/26 | 0.99 |
| −explore | +0.041 | 7/26 | 0.76 |
| −potential (Φ) | +0.004 | 8/26 | 0.75 |
| −reserve (refinement budget) | **+0.116** | 11/26 | **0.012** |
| −two_swap | −0.015 | 1/26 | 0.19 |
| −cascade_aware | 0.000 | 0/26 | 1.0 |

**Table 2.** The gbest guide is the only operator whose removal significantly hurts; the refinement budget reservation significantly hurts and was removed. Caveats, stated honestly: effects are under 1% spread, the effective sample is ~13 graphs, and no multiple-testing correction was applied — the point of this table is selection of a *minimal* configuration, not the magnitude of its components.

**Final configurations.**
- **ABC (proposed)** = CELF-seeded, gbest-guided population search (tournament, directed scouts, dedupe, accept-equal, Φ tie-break, explore_prob 0.15), **no refinement phase**. Elitism + CELF seeding guarantee it never returns worse than CELF.
- **ABC-CELF (hybrid)** = the above plus periodic CELF-style refinement (1-opt + 2-swap) — kept as an ablation because at equal budget it does not earn the "CELF" branding (on WS it is *below* plain ABC in the main test).
- A **Metropolis** variant was tested: on ws1000 k = 30 it recovers +2.5 held-out (66.7 vs 64.3) but remains below SA (73.7); kept as a config flag, not the default.

**The Φ story.** As a fitness term Φ hurt; as a tie-break it is neutral in the ablation; Greedy+Φ underperforms Greedy on WS (23.6 vs 27.2 family mean). Reported as a negative result — one sentence in the write-up.

---

## 5. Evaluation Protocol (v3)

1. **Equal budget.** Greedy's exact cost is k × |pool| fresh evaluations. Every budgeted solver (Greedy, Greedy+Φ, CELF+LS, Sandwich-CELF, SA, GRASP, ABC variants) is capped at that same budget; CELF runs as the cheap baseline. Budget audit: 0 violations across 532 rows of the main study.
2. **Held-out scoring.** Every final seed set is re-scored on fresh, independent threshold draws with R = 1000 (`spread_heldout`); the in-sample value is reported separately. Mean in-sample-minus-held-out gap for ABC: +0.13 (worst +4.4) — overfitting is not driving the conclusions.
3. **OPT bias corrected.** The brute-force optimum is re-scored on the held-out draws; ratios are computed against that held-out OPT.
4. **Graph as the statistical unit.** Spreads are averaged over k within a graph before any test, so correlated k-configurations cannot masquerade as independent evidence. Pairwise differences are tested with Wilcoxon signed-rank across graphs, with cluster-bootstrap (graph-resampling) confidence intervals.
5. **Multiplicity.** Nine family-by-comparison tests ⟹ Bonferroni threshold 0.0056. Exactly one comparison is pre-specified as primary: **ABC vs Greedy on WS** (n = 16 graphs). All other comparisons are exploratory and reported as effect sizes with CIs.
6. **Model definition** (for supervisor sign-off): progressive activation, strict inequality (net > θ·deg, tie handled by the need = floor(θ·deg)+1 convention), frozen thresholds, common random numbers within a model, independent draws across fit/score.

**Estimator.** R = 100–200 for fitting (common random numbers), R = 1000 for held-out scoring; vectorised sparse batched cascades on the RTX 3060 GPU (`backend="auto"`).

**Graphs.** Main study: karate, lesmis, ba500, ba1000, ba2000, ws1000, ws2000, powerlaw1000, powerlaw2000, sbm900. Multi-instance study: 16 × WS(1000), 8 × BA(1000), 8 × SBM(3×300), with generator seeds *and* structural parameters (degree/attachment/density) varied per instance. Tuning graphs: the 13 listed in §4.3. Signed: hub-blocker, signed star/cycle, signed SBM 120/450, signed WS 300, signed BA 200/500.

**Algorithms.** Random, Degree, PageRank, k-Core, MajorityDiscount (zero-MC); BFSBall, DenseCluster (zero-MC constructive, target-set-selection style); Greedy, Greedy+Φ; CELF; CELF+LS (1-opt + 2-swap + cascade-aware — this alone reaches OPT on karate, repairing the coalition 1-opt could not); Sandwich-CELF; SA (1-swap, geometric cooling, reheat, frozen-state early stop); GRASP; ABC (proposed); ABC (no refinement); ABC-CELF (hybrid).

---

## 6. Results I — Exact Optima on Small Graphs (held-out corrected)

Brute force over all k-subsets; OPT itself re-scored on held-out draws (ratio = held-out spread / held-out OPT).

| graph (k) | Degree | Greedy | CELF | CELF+LS | ABC (proposed) |
|---|---|---|---|---|---|
| karate (4) | 0.955 | 0.977 | 0.917 | **1.000** | **1.003** |
| lesmis (3) | 0.744 | **1.000** | 0.872 | 0.884 | **1.000** |
| florentine (4) | 0.937 | **1.000** | 0.833 | 0.937 | **1.000** |
| davis (3) | 0.950 | **1.000** | 0.741 | 0.760 | 0.977 |
| star60 (3) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cycle60 (3) | 0.535 | **1.000** | 1.000 | 1.000 | **1.000** |

**Table 3.** Held-out approximation ratios against the brute-force optimum. ABC reaches the optimum on 4 of 6 graphs (≈ on karate/davis); Greedy on 4; CELF on 2, and CELF is the only method that never beats 0.92 when it loses. (The earlier draft's claim that CELF+1-opt reaches OPT was based on unequal-budget runs and is superseded: at equal budget it reaches OPT on karate only.)

On grid100 (a pathological plateau case) the whole family saturates at ≈81% of OPT at equal budget; ABC reaches 97% with a 17× budget.

---

## 7. Results II — Main Study (10 graphs, equal budget, held-out R = 1000)

### 7.1 Absolute scale: activation fractions

Held-out spread / n, averaged over all (graph, k) configs:

| algorithm | activation fraction | algorithm | activation fraction |
|---|---|---|---|
| SA | 0.2087 | Degree | 0.1785 |
| ABC (proposed) | 0.2080 | CELF | 0.1728 |
| ABC (no refinement) | 0.2085 | Sandwich-CELF | 0.1631 |
| ABC-CELF (hybrid) | 0.2072 | k-Core | 0.1516 |
| Greedy | 0.2061 | Random | 0.0699 |
| Greedy+Φ | 0.2060 | | |
| GRASP | 0.1929 | MajorityDiscount | 0.1799 |
| CELF + LS | 0.1853 | PageRank | 0.1845 |

**Table 4.** The honest scale of the effects: ABC/SA/Greedy differ by <1% of n on average; CELF sits 3.5 points below; the WS wins (Table 6) are real but localised.

### 7.2 Wall-clock and stability

| solver | wall time (s/config) | notes |
|---|---|---|
| CELF | 0.18 | cheap baseline |
| CELF + LS | 0.74 | |
| ABC (proposed) | 4.5 | |
| Greedy | 5.96 | exact k×|pool| evaluations |
| GRASP | 6.04 | |
| ABC-CELF (hybrid) | 7.14 | |
| Sandwich-CELF | 7.92 | runs CELF twice (on σ̄ and σ) |
| SA | 10.71 | 2.5 ms/eval vs ABC 1.9 ms/eval — cost is evaluations, not overhead |

**Table 5.** ABC is the cheapest of the set-level methods; SA's extra time is genuine search, not overhead. **Equal-wall-clock secondary analysis** (ABC at ≈2.4× budget = SA's wall time): on ws1000 k = 15, ABC stays at 37.5 while SA at 1× budget reaches only 30.7 — ABC wins the equal-time comparison (see §7.6). ABC run-to-run std: 1.05 spread units (CV 1.7%).

### 7.3 CELF breaks — audit (repeated for emphasis)

Lazy violations 50.24% (418/832); differs from Greedy 16/17; strictly worse 13/17; one near-tie.

### 7.4 Per-configuration ABC-minus-Greedy gains (held-out, %)

| config | % | config | % | config | % | config | % |
|---|---|---|---|---|---|---|---|
| ws1000 k10 | +98.3 | ws2000 k10 | +50.8 | sbm900 k10 | +14.9 | ba2000 k5 | +13.2 |
| ws2000 k5 | +19.3 | ws1000 k5 | +16.4 | ws1000 k20 | +13.8 | sbm900 k5 | +12.3 |
| lesmis k10 | +5.4 | powerlaw2000 k5 | +5.8 | powerlaw1000 k5 | +4.4 | ws1000 k30 | +3.7 |
| ws2000 k20 | +8.3 | sbm900 k20 | +3.4 | ws2000 k30 | +3.5 | ba500 k5 | +1.5 |
| ba500 k20 | +1.4 | lesmis k5 | +0.9 | sbm900 k30 | +1.6 | powerlaw2000 k20 | +0.7 |
| powerlaw1000 k10 | +0.4 | ba2000 k10 | +0.2 | ba1000 k30 | +0.1 | karate k10 | 0.0 |
| lesmis k20/30, karate k5 | ≈0 | ba500 k10 | +0.1 | ba500 k30 | −0.6 | ba1000 k5 | −0.4 |
| ba1000 k10 | −2.9 | powerlaw1000 k20 | −0.5 | powerlaw2000 k10 | −1.0 | powerlaw2000 k30 | −2.3 |
| ba1000 k20 | −0.8 | ba2000 k20 | −0.3 | ba2000 k30 | 0.0 | powerlaw1000 k30 | −0.8 |

**Table 6.** The advantage is a **WS/SBM phenomenon** (wins up to +98%), a wash on BA/powerlaw (median ≈ 0.03%, 11/20 configs positive), and near-zero on the small graphs. Note ws1000 k5: Greedy's spread is 5.0 (no cascade at all) — that +16% is 1.4 nodes out of 1,000; the *mechanism* is interesting, the absolute number is small.

---

## 8. Results III — Multi-Instance Family Study (graph = unit)

### 8.1 Primary comparison: ABC vs Greedy on WS, n = 16 independent instances

16 fresh Watts–Strogatz(1000) instances, degree ∈ {6, 8, 10}, varying seeds; spread averaged over k ∈ {5, 10, 20} within each graph; paired differences across graphs with cluster-bootstrap CIs (`results/s10_ws16_stats.json`, forest plot `figures/fig8_forest_WS16.png`).

| comparison | mean diff | wins | Wilcoxon p | bootstrap CI |
|---|---|---|---|---|
| **ABC − Greedy (primary)** | **+4.57** | 14/16 | **3.1 × 10⁻⁴** | **[+2.46, +6.91]** |
| SA − Greedy | +5.99 | 15/16 | 1.5 × 10⁻⁴ | [+4.01, +7.85] |
| ABC − SA | −1.41 | 6/16 | 0.093 | [−2.84, −0.12] |
| ABC − CELF | +2.69 | 16/16 | 3.1 × 10⁻⁵ | [+1.83, +3.89] |

**Table 7.** The primary comparison passes Bonferroni (0.0056 over 9 tests). SA ≥ ABC on WS (CI excludes 0); both ≫ Greedy; both ≫ CELF.

Family means: SA 26.0, ABC 24.6, CELF+LS 24.1, CELF 21.9, Greedy 20.0, BFSBall 13.6, DenseCluster 12.9.

Per-graph ABC−Greedy differences: +13.8, +11.8, +11.7, +6.8, +6.4, +6.0, +4.5, +4.4, +2.6, +2.1, +2.0, +1.5, +1.0, +0.03, −0.7, −0.8 (14 of 16 positive).

**Cascade metric (draws-based).** Fraction of the 1,000 held-out threshold draws in which the final cascade exceeds k + 10: ABC 0.542, SA 0.509, CELF 0.428, **Greedy 0.330** (≥ k + 5: 0.696 / 0.680 / 0.614 / 0.484). This replaces the earlier "spread > k" definition, which a tiny excess could satisfy.

### 8.2 BA family (8 instances): the wash

ABC − Greedy = +0.26 (5/8, p = 0.64, CI [−0.42, +1.06]) — a wash. ABC − SA = −0.49 (2/8, p = 0.039, CI [−0.79, −0.18]) — SA favoured. ABC − CELF = +12.4 (8/8, p = 0.0078). Means: SA 40.6, ABC 40.1, Greedy 39.8, CELF+LS 30.3, CELF 27.7, DenseCluster 26.7, BFSBall 22.9.

### 8.3 SBM family (8 instances): suggestive

ABC − Greedy = +0.58 (6/8, p = 0.078, CI [+0.07, +1.14]) — call it **suggestive** (CI excludes 0, p is weak at n = 8). ABC − SA = −0.05 (tie). ABC − CELF = +1.57 (8/8, p = 0.0078).

### 8.4 Constructive baselines: search matters, not just locality

SA − BFSBall on WS: +13.1 (8/8, p = 0.0078, CI [+12.1, +14.1]); on BA +17.6 (8/8); on SBM +2.0 (8/8). Dense connected coalitions are *necessary* but nowhere near *sufficient*: the evaluation budget spent by set-level search is what buys the cascade.

### 8.5 Regime: the contagion-threshold window

Let k* = the smallest tested k at which **Greedy's** final set first triggers any cascade. On WS the ABC advantage concentrates below k*: **+64.6% relative mean below k* vs +23.0% at/above** (d_k = −10: +84% / +8.4 nodes; d_k = +15: +7.5%). On BA, Greedy already cascades at k = 5 (hubs) → no window → wash. On SBM the same pattern, weaker (+16.7% vs +3.4%).

**Circularity caveat (stated in the text):** defining k* from Greedy makes this a description of *Greedy's blindness*, not an independent mechanism. An independent percolation definition failed constructively: for k ≤ 20, **no random k-set ever triggers a cascade** on any instance — the very fact chosen coalitions are required *is* the phenomenon.

### 8.6 Budget scaling and the ceiling question (S9)

| graph | budget mult | Greedy | CELF | ABC | SA |
|---|---|---|---|---|---|
| ba500 | 0.5× | 92.85 | 52.48 | 93.30 | 92.34 |
| ba500 | 1× | 92.85 | 52.48 | 93.30 | 92.85 |
| ba500 | 4× | 92.85 | 52.48 | 93.95 | 93.07 |
| ba500 | 8× | 92.85 | 52.48 | 93.95 | 93.95 |
| ws1000 | 0.5× | 27.49 | 29.79 | **36.13** | 35.17 |
| ws1000 | 1× | 27.49 | 29.79 | **37.45** | 30.66 |
| ws1000 | 4× | 27.49 | 29.79 | 37.45 | 37.59 |
| ws1000 | 8× | 27.49 | 29.79 | 37.45 | 35.59 |

**Table 8.** Held-out spread vs budget (k = 15; Greedy's cost is budget-fixed, hence its flat line). ABC reaches its ceiling at 0.5–1×; SA needs 4× to arrive at the *same* ceiling.

**Ceiling checks.** (a) ABC at 8× consumes all 24,000 evaluations (verified) — the flatness is a plateau, not early stopping. (b) A 16× multi-start SA run: best 38.72 vs 37.59 at 4× — no escape. (c) The sandwich bound is vacuous at this scale (OPT ≤ 1495). Conclusion: both methods sit at a shared ceiling; near-optimality is supported only by budget-insensitivity (a 32× budget increase buys < 1 spread unit) — the phrase "escapes the basin" is *not* supported, and "shared ceiling" is the defensible wording. On ba500 the ceiling ≈ 94 ≈ Greedy's 92.9 from 0.5× — greedy is already near-optimal there, consistent with the §8.2 wash.

### 8.7 The δ-sweep — and its confound

| δ (ba500) | γ_min | viol | CELF | Greedy | ABC | vs CELF | vs Greedy |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.56 | 70% | 46.0 | 69.0 | 71.0 | +54% | +2.9% |
| 0.05 | 0.52 | 73% | 60.7 | 85.1 | 85.6 | +41% | +0.6% |
| 0.10 | 0.52 | 74% | 61.6 | 86.9 | 85.4 | +39% | −1.7% |
| 0.15 | 0.53 | 76% | 66.1 | 92.8 | 92.3 | +40% | −0.5% |
| 0.20 | 0.68 | 70% | 101.2 | 114.1 | 112.0 | +11% | −1.9% |
| 0.30 | 0.82 | 68% | 231.0 | 239.6 | 237.5 | +3% | −0.9% |
| 0.40 | 0.80 | 55% | 289.8 | 293.2 | 291.7 | +1% | −0.5% |
| 0.50 | 0.95 | 0% | 297.0 | 297.0 | 298.0 | +0.4% | +0.4% |

| δ (ws1000) | γ_min | CELF | Greedy | ABC | vs Greedy |
|---|---|---|---|---|---|
| 0.00 | 1.00 | 27.0 | 16.0 | 28.0 | **+75%** |
| 0.05 | 1.00 | 27.0 | 25.5 | 32.5 | +27% |
| 0.10 | 0.92 | 26.6 | 25.2 | 34.0 | +35% |
| 0.15 | 0.80 | 30.3 | 27.5 | 34.2 | +24% |
| 0.20 | 0.72 | 31.2 | 23.3 | 35.7 | +53% |
| 0.30 | 0.67 | 35.6 | 37.7 | 37.3 | −1.2% |
| 0.40 | 0.77 | 34.1 | 46.9 | 45.3 | −3.4% |
| 0.50 | 0.99 | 103.3 | 103.5 | 103.4 | −0.1% |

**Table 9.** On ba500 the ABC-vs-CELF gap shrinks monotonically with δ (Spearman ρ = −0.976, p = 3 × 10⁻⁵); the ABC-vs-Greedy gap is a wash at every δ (ρ = −0.29, p = 0.49). On ws1000 the ABC-vs-Greedy advantage is large for δ ≤ 0.2 and gone beyond.

**Confound diagnosis (replaces the earlier "tracks submodularity" claim).** Both γ and the violation rate move with δ, so raw correlations are near-automatic. Controlling for δ by partial correlation: every association vanishes — ba500 vs-Greedy ρ_partial = +0.42 (p = 0.31); vs-CELF ρ_partial = +0.13 (p = 0.75); ws1000 vs-Greedy ρ_partial = +0.03 (p = 0.95). A fixed-δ test across the 24 multi-instances (violation rate vs ABC-minus-Greedy gain, per graph) is also null: ρ = −0.23 (p = 0.28). **Statement for the report:** the advantage is a plateau/cascade-threshold phenomenon (Table 9, ws1000), and no reliable association with the measured violation rate survives control — the structural claims rest on §2 and the CELF audit, not on the sweep.

---

## 9. Results IV — Signed Graphs

### 9.1 Model and non-monotonicity

Signed, weighted extension (§1.1). Negative edges break monotonicity: a **constructive witness** — hub with 4 positive leaves, a positive follower and one enemy leaf — σ({1,2,3,4}) = 6 > σ({1,2,3,4,5}) = 5 (seeding the enemy blocks the hub). A **located witness** on a random signed BA graph (200 nodes, p_neg = 0.2): σ(S) = 48.67, σ(S ∪ {v}) = 48.42, drop 0.26 (above the 4-SE tolerance), found by structured search (S = positive neighbourhood of a node just below its threshold, v = its enemy). Random pair sampling almost never sees these (1/400 on the hub example; 0/2000 elsewhere) — the violation is a needle, which is why the structured search exists.

### 9.2 Per-configuration held-out spreads

| graph | k | Greedy | CELF | CELF+LS | SA | ABC |
|---|---|---|---|---|---|---|
| signed_sbm120 | 15 | 20.4 | 18.2 | 18.2 | **44.0** | 40.6 |
| signed_sbm120 | 10 | 10.6 | 10.2 | 10.4 | 13.2 | 12.2 |
| signed_ws300 | 15 | 21.7 | 22.2 | 24.7 | 26.6 | **27.2** |
| signed_ws300 | 10 | 13.9 | 14.6 | 15.1 | 15.3 | 16.7 |
| signed_ba500 | 15 | 28.0 | 22.0 | 23.7 | **29.3** | 28.8 |
| signed_ba500 | 10 | 19.2 | 15.1 | 15.5 | 18.4 | 17.9 |
| signed_cycle40 | 15 | 30.9 | 30.9 | 30.9 | 31.0 | 30.9 |

**Table 10.** Family means: SA 17.7 ≈ ABC 17.6 > Greedy 15.4 > CELF+LS 15.1 > CELF 14.6 > NetSupport 11.8. Where cascades exist (sbm120 k = 15: ABC/SA ≈ 2× Greedy), set-level search wins; on the BA signed graph it is a near-tie with Greedy. (signed_sbm450 activates nothing at k ≤ 15 — a trivial regime reported as such.)

### 9.3 Sign-ablation control — the decisive negative result

Identical topology, all-positive twin, same protocol (`results/s11_signcontrol.csv`):

| graph | k | edges | Greedy | ABC | ABC vs Greedy |
|---|---|---|---|---|---|
| sbm120 | 15 | signed | 19.9 | 38.9 | +95% (+19.0 nodes) |
| sbm120 | 15 | positive twin | 19.8 | 55.6 | **+181% (+35.8 nodes)** |
| ws300 | 15 | signed | 21.0 | 26.9 | +27.6% (+5.8) |
| ws300 | 15 | positive twin | 36.8 | 40.7 | +10.8% (+4.0) |

**Table 11.** The ABC-vs-Greedy advantage **persists — even grows — when the signs are removed**. It is the plateau effect, not the sign structure. Signed contribution = model generalisation + non-monotonicity witness; no distinct search advantage. (Negative edges do shrink everyone's absolute spread — compare ws300 rows — but that is a model property, not a search effect.)

---

## 10. Corrections Relative to Earlier Drafts

| earlier claim | corrected, with evidence |
|---|---|
| "89% lazy-violation rate" | **50.24%** over all CELF runs (418/832); 89% was 8/9 checks in one smoke test |
| "+46.5% mean / +21.6% median / +273% max over CELF" | replaced by held-out, equal-budget comparisons: per-family CIs (Table 7) and per-config gains (Table 6). The +273% is real but is CELF collapsing (131.9 vs Degree 473.7, ba500 k=30) |
| "γ bound 0.18–0.36" | measured γ_min 0.292–0.845 across majority settings (Table 1) — and it is an optimistic estimate |
| "violation for every δ < 0.5" (K₁,₃) | K₁,₃ clear only for δ ≲ 0.3; larger stars (K₁,₉) extend it to δ = 0.40 (§2.2) |
| Mossel–Roch "iff" | sufficiency only; code and figures corrected (§2.3) |
| "Karaboga (2007) ABC" | Karaboga 2005 TR06; Karaboga & Basturk 2007 (§4.2, references) |
| Lu et al. hardness for uniform-½ | their hardness is for general deterministic thresholds; does not transfer automatically (§2.5) |
| "CELF+1-opt reaches OPT on small graphs" | unequal-budget artefact; at equal budget it reaches OPT on karate only; ABC reaches it on 4/6 (Table 3) |
| "2,000+ monotonicity pairs across 18 configs" | 5,800 pairs: 600/600/250 × 12 configs, 0 violations |
| "ABC-CELF (proposed)" as the method name | CELF refinement does not earn its budget (LOO Table 2; hybrid < plain ABC on WS) — method renamed **ABC (proposed)**; hybrid kept as ablation |
| "advantage tracks the loss of submodularity" (δ-sweep) | confounded; no association survives control (§8.7). Structural claims rest on §2 + the CELF audit |
| "signed graphs give ABC a distinct advantage" | sign control shows the advantage is the plateau (§9.3) |
| trivial-baseline-relative headlines | replaced by activation fractions (Table 4), per-config absolute+relative gains (Table 6), draws-based cascade metric (§8.1) |

---

## 11. Discussion and Limitations

1. **ABC is a valid set-level searcher, not the best one.** SA ≥ ABC on WS and BA. ABC's defensible properties: budget efficiency (ceiling at 1× vs SA's 4×, Table 8), the ≥CELF guarantee, run-to-run stability (CV 1.7%), and 6× cheaper wall-clock at equal quality. If the assignment requires ABC, this framing is strong and credible.
2. **Small absolute effects.** Activation fractions differ by <1% of n among ABC/SA/Greedy (Table 4); the WS wins are +2.5…+8.4 nodes on n = 1,000. Report CIs and forest plots, not percentages against trivial baselines.
3. **Statistics.** Primary comparison Bonferroni-safe at n = 16; BA/SBM exploratory at n = 8 (SBM "suggestive"). More instances per family would firm these up — the harness supports it (append `s10_multi.csv`, resume works).
4. **Ceiling certification** is open: the sandwich bound is vacuous at this scale; near-optimality rests on budget-insensitivity alone. A small-scale exact check or a tighter bound would close this.
5. **No real-network evaluation yet** (ego-Facebook, email-Eu-core; Epinions/Slashdot/Bitcoin-Alpha for signed) — flagged as future work; the signed loader (`load_signed_edgelist`) is in place.
6. **Candidate pool** caps at 200 nodes; both ABC and CELF search the pool, so neither is guaranteed the unconstrained global optimum on large graphs (exact verification was restricted to small graphs, Table 3).
7. **k choice** is fixed at {5, 10, 20, 30}; the regime analysis (§8.5) suggests reporting relative to a graph's contagion threshold in future work.
8. **Negative results recorded:** Φ as a fitness term; Greedy+Φ on WS; the refinement reserve; the Metropolis default; the δ-sweep correlation; signed-as-distinct-advantage.

---

## 12. Conclusions

1. **The model is monotone and not submodular.** Monotonicity is proved (lock-step induction) and machine-checked (5,800 pairs, 0 violations). Non-submodularity is witnessed by the K₁,₃ counterexample, explained by the Mossel–Roch sufficient condition (CDF concavity), and certified by exhaustive scans: 23–76% of non-degenerate triples violate diminishing returns on majority variants, versus 0.0% for the LT control.
2. **CELF breaks, measurably and decision-relevantly.** 50.2% lazy-bound violations; different from Greedy in 16/17 configs, strictly worse in 13/17; on ba500 k = 30 it collapses to a quarter of Greedy's and Degree's spread.
3. **Set-level search beats marginal search on sparse clustered graphs, below the contagion threshold.** On 16 independent WS instances, ABC − Greedy = +4.6 held-out (14/16, CI [+2.5, +6.9], p = 3 × 10⁻⁴, Bonferroni-safe); SA − Greedy = +6.0 (15/16). The advantage concentrates below Greedy's cascade threshold k*, and no random k-set cascades for k ≤ 20 — cascades require chosen coalitions. On hub-heavy BA graphs both tie Greedy; both beat CELF everywhere; constructive locality baselines (BFS balls, dense clusters) are far behind (search, not just locality, is what matters).
4. **SA ≥ ABC** — the claim "ABC is the best set-level searcher" is not supported and should not be made. ABC is a valid, budget-efficient, guarantee-carrying member of the class.
5. **Signed extension** generalises the model and exhibits non-monotonicity (constructive + located witnesses), but confers no distinct search advantage (sign-ablation control).

**Recommended framing for submission:** prove monotonicity; disprove submodularity and identify the structural reason; quantify γ as the honest replacement for (1−1/e); document CELF's failure with the audit; and present ABC — motivated by the literature on ABC's weaknesses and validated by a leave-one-out ablation — as an effective, budget-efficient set-level optimiser for a model where coalitions, not individuals, carry the influence.

---

## References

- Akay, B. & Karaboga, D. (2012). A modified artificial bee colony algorithm for real-parameter optimization. *Information Sciences*, 192, 120–142.
- Cantini, R., Marozzo, F., Mazza, S., Talia, D. & Trunfio, P. (2021). A Weighted Artificial Bee Colony Algorithm for Influence Maximization. *Online Social Networks and Media*.
- Das, A. & Kempe, D. (2011). Submodular meets spectral: greedy algorithms for subset selection, sparse approximation and dictionary selection. *ICML*.
- Karaboga, D. (2005). An Idea Based on Honey Bee Swarm for Numerical Optimization. *Technical Report TR06*, Erciyes University.
- Karaboga, D. & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. *Journal of Global Optimization*, 39(3), 459–471.
- Kempe, D., Kleinberg, J. & Tardos, É. (2003). Maximizing the spread of influence through a social network. *KDD*.
- Kiran, M. S. & Fındık, O. (2015). A directed artificial bee colony algorithm. *Applied Soft Computing*, 26, 454–462.
- Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J. & Glance, N. (2007). Cost-effective outbreak detection in networks. *KDD*.
- Lu, W., Chen, W. & Lakshmanan, L. V. S. (2015). From competition to complementarity: comparative influence diffusion and maximization. *PVLDB*.
- Lu, Z., Zhang, W., Wu, W., Kim, J. & Fu, B. (2012). The complexity of influence maximization problem in the deterministic linear threshold model. *Journal of Combinatorial Optimization*, 24(3), 374–378.
- Mossel, E. & Roch, S. (2007). On the submodularity of influence in social networks. *STOC*.
- Nemhauser, G. L., Wolsey, L. A. & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions—I. *Mathematical Programming*, 14, 265–294.
- Pan, Q.-K., Tasgetiren, M. F., Suganthan, P. N. & Chua, T. J. (2011). A discrete artificial bee colony algorithm for the lot-streaming flow shop scheduling problem. *Information Sciences*, 181(12), 2455–2468.
- Zhu, G. & Kwong, S. (2010). Gbest-guided artificial bee colony algorithm for numerical function optimization. *Applied Mathematics and Computation*, 217(7), 3166–3173.
- "Artificial bee colony meets variable neighborhood search" for influence maximization (2026), *Swarm and Evolutionary Computation* — the critique ("rigid neighbourhood generation, weak directional guidance") motivating W7–W9.
- Zahoor, A., Bashir, J. & Gillani, I. A. (2026). RASH: signed, susceptibility-aware diffusion with γ-weak submodularity. *arXiv*.
- Li, D. et al. (2014). Polarity-related influence maximization in signed social networks. *PLOS ONE* 9(7).

---

## Appendix A — Reproducibility and File Map

**Code.** `majority_rule_model.py` (model, GPU backend, pools with random tail, greedy/greedy_phi/CELF, BFSBall/DenseCluster, heldout_model), `signed_rule_model.py` (signed model, generators, non-monotonicity witness search), `theory_checks.py` (monotonicity, counterexamples, exhaustive scans, γ), `abc_celf.py` (ABC + local search, all W1–W9 flags), `baselines.py` (SA, GRASP, sandwich-CELF), `run_experiments.py` (S1–S11 + figures + summary), `run_validation.py` (OPT, leave-one-out, CELF audit, signed validation), `extend_ws.py` (WS 8→16 instances).

**Runs.** `python run_experiments.py <s1|s2|s5|s5:ws|s8|s9|s10|s11|fig>`; `python run_validation.py <opt|loo|audit|signed>`. All long studies append per-cell and resume.

**Key outputs.** `results/s10_ws16_stats.json` (primary comparison), `results/s10_multi_stats.json` (families at n = 8), `results/summary.json` (S3 + signed + wall-clock + cascade success), `results/s1_structure.json` (theory layer), `results/val_loo.csv`, `results/val_celf_audit.csv`, `results/s11_signcontrol.csv`, `results/s9_budget.csv`, `results/s5_delta_ba500.csv`, `results/s5_delta_ws1000.csv`, `results/s2_exact.csv`, `results/s8_signed.csv`.

**Figures.** `fig1_spread_vs_k` (held-out spread vs k), `fig2_improvement`, `fig3_convergence`, `fig4_delta_sweep`, `fig5_exact_ratio` (held-out OPT ratios), `fig6_structure` (local activation functions + violation/γ), `fig7_budget` (budget scaling), `fig8_forest_WS16` (the per-graph forest plot for the primary comparison), `fig8_forest_BA`, `fig8_forest_SBM`.

## Appendix B — Suggested Slide Deck Logic

1. Model + local activation function (fig6 left).
2. Theory: monotone (proof) / not submodular (K₁,₃ counterexample + Table 1).
3. Why greedy/CELF fail: the plateau; CELF audit (16/17, 50.2%).
4. ABC design: canonical weaknesses → W1–W9 operators → LOO-selected minimal config (Table 2).
5. Methodology in one slide: equal budget, held-out R = 1000, graph = unit, pre-specified primary, Bonferroni.
6. Primary result: WS forest plot (fig8) + Table 7.
7. Regime: gain vs k − k* (§8.5) + the percolation finding.
8. Budget/wall-clock: Table 8 + ceiling checks + equal-time note.
9. Honest caveats slide: SA ≥ ABC; signed = plateau (Table 11); δ-sweep confound; Φ negative.
10. Signed extension: model + witnesses + sign control.
11. Conclusions (the five claims of §12).