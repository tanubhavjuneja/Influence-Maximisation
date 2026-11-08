# Artificial Bee Colony for the Majority-Rule Influence Model — Final Findings

Everything in this document is reproducible from the code and CSVs in this folder.
All headline numbers are **equal-budget** (shared budget = k × |pool| = Greedy's exact cost,
audited: 0 violations across all runs) and **held-out** (final seed sets re-scored on
fresh threshold draws, R = 1000). The **graph is the statistical unit**: spreads are
averaged over k within a graph, then compared across independent instances with
Wilcoxon signed-rank tests and cluster-bootstrap confidence intervals.

---

## 1. The model

Progressive (once active, always active) threshold diffusion on an undirected graph.
Each node v draws a susceptibility threshold θ_v once and keeps it:

```
v activates  iff   |{active neighbours of v}|  >  θ_v · deg(v)
```

- `dmr` (deterministic majority): θ = ½ always.
- `smr` (stochastic majority): θ ~ U[½−δ, ½+δ]; δ = 0.15 is the workhorse.
- `lt` (Linear Threshold, Kempe et al. 2003): θ ~ U[0,1] — the submodular control.
- `slt` (sandwich upper bound): θ ~ U[0, ½+δ] — see §3.

σ(S) = expected number of finally-active nodes, estimated by Monte-Carlo over R
pre-sampled, frozen threshold vectors (common random numbers inside a model; a
separate held-out model for scoring). Signed extension: edge signs/weights, node v
activates iff (positive active weight − negative active weight) > θ_v · deg(v).

## 2. Structural results (S1 — theory_checks.py)

**Monotone — proven and machine-checked.** Lock-step induction on the cascade:
S ⊆ T ⟹ A_t(S) ⊆ A_t(T) for all t, pointwise in θ, so σ(S) ≤ σ(T).
Empirically: 5,800 sampled pairs (600/600/250 per configuration × 12 configurations),
**0 violations**.

**NOT submodular — machine-checked counterexamples.**
- Minimal witness K₁,₃ (star, 3 leaves): gain of y to {} is 1, gain of y to {x} is 3
  (x, y flip the centre, the centre flips z) — increasing returns.
- Stochastic robustness: on K₁,₃ the violation survives δ ≲ 0.3; on K₁,₉ (T = 4 leaves)
  it is still visible at δ = 0.35 (+0.40) and δ = 0.40 (+0.16), gone by δ = 0.45.
- Exhaustive scans (max_T = 3): under smr δ = 0.15, 23–52% of triples violate
  submodularity (60–76% of non-degenerate triples). Under LT: 0.0%.
- Submodularity ratio γ_min (Das & Kempe 2011): **0.29–0.85** across majority settings
  (LT control 0.95–0.99). An optimistic estimate: the true minimum is at most this.

**Mossel & Roch (2007)**: submodular local activation functions are a **sufficient**
condition for σ's submodularity (not iff). The majority CDF is an S-curve — not concave —
so the sufficient condition fails; the actual violations are then certified empirically.

**Hardness**: Lu, Zhang, Wu, Kim & Fu (J. Comb. Optim. 24(3):374–378, 2012) prove no
poly-time n^(1−ε)-approximation for the deterministic threshold model with *general*
thresholds. Our uniform-½ rule is a special case, so their hardness does not transfer
automatically; approximation quality is measured against the brute-force optimum instead.

## 3. The sandwich bound (Lu, Chen & Lakshmanan 2015 style)

The concave envelope of the majority CDF on [0,1] is **F̄(x) = min(x/(½+δ), 1)**,
i.e. the `slt` model with θ ~ U[0, ½+δ]. Because F̄ ≥ F pointwise, a common-uniform
coupling gives **σ̄ ≥ σ pathwise**, and F̄ concave ⟹ **σ̄ is submodular** (Mossel–Roch),
so CELF is *valid* on σ̄. This yields an upper bound OPT ≤ σ̄(S)·e/(e−1) and an
after-the-fact guarantee σ(S) ≥ (1−1/e)·OPT/α(S) with α(S) = σ̄(S)/σ(S).
**Measured**: α ≈ 13 on ba500 — the guarantee is vacuous at small δ (as predicted),
and σ̄'s OPT bound is loose (e.g. ≤ 1495 on ws1000 where the true optimum is ~38).
Kept as the theory layer; Sandwich-CELF is the weakest budgeted seeder.

## 4. Why greedy/CELF fail, and the ABC design

Forward greedy (hence CELF) is driven by singleton marginal gains. Under a majority
rule those gains are ~zero until a coalition reaches critical mass, so greedy is blind
on the plateau. ABC searches whole k-sets with swap operators, which can assemble and
*discard* coalitions (removing harmful seeds on signed graphs).

Canonical ABC (Karaboga 2005 TR06; Karaboga & Basturk 2007) weaknesses, and what the
literature says about them:

| # | weakness | literature fix | ours |
|---|---|---|---|
| W1 | weak exploitation | gbest-guided ABC (Zhu & Kwong 2010) | gbest partner with prob 0.6 |
| W2 | premature convergence | tournament onlookers (vs roulette) | tournament, size 4 |
| W3 | blind scout restarts | directed scouts (Kiran & Fındık 2015) | rebuild near incumbent |
| W4 | fixed perturbation | modification rate (Akay & Karaboga 2012) | n_mod swaps + mutation |
| W5 | duplicate sources | — | dedupe |
| W6 | refinement blows budget | — | remaining-budget-only + gating |
| W7 | rigid moves on plateaus (ABC+VNS critique) | — | Φ tie-break + accept-equal + random exploration moves |
| W8 | 1-opt can't repair coalitions | — | cascade-aware candidates + 2-swap shaking |

**Final configuration** (chosen by leave-one-out ablation on 13 disjoint tuning graphs ×
26 configs × 20 seeds, `results/val_loo.csv`): gbest-guide is the only operator whose
removal significantly hurts (−0.13 spread units, p = 0.0006); the refinement reserve
significantly hurts when present (+0.12 to remove, p = 0.012); everything else is
neutral (effects < 1% — reported honestly). So:

- **ABC (proposed)** = CELF-seeded, gbest-guided population search, no refinement phase.
  Elitist ⟹ never worse than CELF.
- **ABC-CELF (hybrid)** = + CELF-style refinement, kept as an ablation (it does not
  earn the "CELF" branding — the hybrid is *below* plain ABC on WS in the main test).
- A **Metropolis** acceptance variant was tested (+2.5 on ws1000 k=30) — still below SA;
  kept as a config flag, not the default.
- **Φ(S) = Σ_v min(1, net_v/need_v)** at cascade close (plateau gradient, computed in the
  same cascade pass, zero extra cost): as a global fitness term it *hurt* (BA 51.9 vs
  57.5 held-out); as a **tie-breaker within Monte-Carlo noise** it is neutral; Greedy+Φ
  is worse than Greedy on WS (23.6 vs 27.2). Report as a negative result.

## 5. Baselines

Zero-MC: Degree, PageRank, k-core, MajorityDiscount, **BFSBall** (BFS ball from the
top-degree node), **DenseCluster** (iteratively densest connected set) — both
target-set-selection style. Marginal: Greedy, Greedy+Φ, **CELF** (cheap baseline),
**CELF+LS** (1-opt + 2-swap + cascade-aware refinement — this reaches OPT on karate,
fixing a broken coalition 1-opt could not). Set-level: **SA** (1-swap, λ-fitness,
geometric cooling, reheat, frozen-state early stop), GRASP, **ABC (proposed)**,
ABC (no refinement), ABC-CELF (hybrid), Sandwich-CELF.

## 6. Headline results — 16 fresh WS instances (pre-specified primary comparison)

Each graph averaged over k ∈ {5, 10, 20} first; 16 independent Watts–Strogatz(1000)
instances with degree ∈ {6, 8, 10} and fresh seeds. Paired differences per graph are in
`figures/fig8_forest_WS16.png` and `results/s10_ws16_stats.json`.

| comparison | mean diff (held-out) | wins | Wilcoxon p | bootstrap CI |
|---|---|---|---|---|
| **ABC vs Greedy (primary)** | **+4.57** | 14/16 | **3.1×10⁻⁴** | **[+2.46, +6.91]** |
| SA vs Greedy | +5.99 | 15/16 | 1.5×10⁻⁴ | [+4.01, +7.85] |
| ABC vs SA | −1.41 | 6/16 | 0.093 | [−2.84, −0.12] |
| ABC vs CELF | +2.69 | 16/16 | 3.1×10⁻⁵ | [+1.83, +3.89] |

- Family means: SA 26.0, ABC 24.6, CELF+LS 24.1, CELF 21.9, Greedy 20.0,
  BFSBall 13.6, DenseCluster 12.9.
- **Cascade fraction** (share of held-out threshold draws with spread ≥ k+10):
  ABC 0.54, SA 0.51, CELF 0.43, **Greedy 0.33**. (≥ k+5: 0.70/0.68/0.61/0.48.)
- With 9 family-by-comparison tests the Bonferroni threshold is 0.0056; the primary
  comparison passes (p = 3×10⁻⁴). All other comparisons are exploratory: report CIs,
  not p-values. Per-graph paired differences: ABC−Greedy positive on 14/16
  (range −0.8 … +13.8); ABC−SA positive on only 6/16.

**BA (8 instances)** — wash vs Greedy (+0.26, 5/8, p = 0.64, CI [−0.42, +1.06]);
SA ≥ ABC (−0.49, 2/8, p = 0.039, CI [−0.79, −0.18]); ABC ≫ CELF (+12.4, 8/8).

**SBM (8 instances)** — ABC vs Greedy +0.58, 6/8, CI [+0.07, +1.14], p = 0.078 →
call it *suggestive*; ABC ≈ SA; ABC > CELF (+1.57, 8/8).

**Locality is not the story**: SA ≫ BFSBall/DenseCluster on every family
(WS: 26.0 vs 13.6/12.9, p = 0.008) — set-level *search* matters, not just connected
coalitions.

## 7. Regime: the contagion-threshold window (S10)

Define k* = smallest tested k at which **Greedy first triggers any cascade** (held-out
spread > k). On WS the ABC advantage concentrates **below k***: mean relative gain
+65% below k* vs +23% at/above; absolute gains +2.5 … +8.4 nodes (largest at k*−10).
Above k*+15 the advantage is ≈ +7%. On BA, Greedy already cascades at k = 5 (hub graphs)
→ no window → wash. SBM shows the same pattern, weaker (+17% vs +3%).

Honesty note: this is a description of greedy's blindness, not an independent mechanism.
An attempt to define k* independently by percolation (random k-sets) failed because
**random sets never cascade at k ≤ 20** — the very fact optimized search does. That is
itself the phenomenon: majority cascades require *chosen* coalitions.

## 8. What does NOT hold (negative results, reported as such)

- **δ-sweep "advantage tracks non-submodularity" — confounded.** Partial Spearman
  correlation controlling for δ kills every association (p ≥ 0.31). A fixed-δ test
  across the 24 multi-instances also shows no reliable correlation between the
  submodularity-violation rate and the ABC-vs-Greedy gain (ρ = −0.23, p = 0.28).
  The structural claims rest on §2 and the CELF audit, not on the sweep.
- **Signed graphs confer no distinct search advantage.** Sign-ablation control
  (`results/s11_signcontrol.csv`): the ABC-vs-Greedy advantage persists — and even
  grows — on the all-positive twin of the same topology (sbm120 k=15: +181% positive
  vs +95% signed; absolute +35.9 vs +19.0 nodes; ws300: +10.8% positive vs +27.6%
  signed). The effect is the plateau, not the signs. The signed model contributes a
  generalisation and a **non-monotonicity witness** (hub-blocker example; a drop-0.26
  witness found on signed_ba200), not a new search advantage.
- **Φ** as fitness: hurts. Greedy+Φ: worse than Greedy on WS. One sentence each.
- **CELF refinement in the hybrid**: does not pay at equal budget; demoted.
- **ABC is not the best set-level searcher** — SA ≥ ABC on WS (CI excludes 0) and BA
  (p = 0.039). ABC is a *valid* set-level searcher, cheapest at equal quality
  (see §9), with a ≥CELF guarantee and model-aware operators.

## 9. Budget and wall-clock (S9)

- **ba500 k=15**: Greedy 92.9; ABC/SA ≈ 93 from 0.5× budget; CELF stuck at 52.5 —
  all at the shared ceiling.
- **ws1000 k=15**: ABC reaches its ceiling 37.5 at **1× budget** and stays flat through
  8× (it spends the full budget — verified 24,000/24,000 evaluations — so the flatness
  is a plateau, not early stopping); SA needs 4× to reach the same ceiling (37.6); a
  16× multi-start SA gets 38.7 — no escape. Both methods sit at the same shared
  ceiling; the sandwich bound is vacuous at this scale, so near-optimality is supported
  only by budget-insensitivity (32× more budget buys < 1 spread unit).
- **Wall-clock per config** (S3): CELF 0.18 s, CELF+LS 0.74 s, ABC 4.5 s, Greedy 6.0 s,
  SA 10.7 s. SA's extra time is evaluations, not overhead (2.5 vs 1.9 ms/eval).
  **Equal-wall-clock secondary**: ABC at SA's wall time (≈2.4× budget) stays at 37.5
  vs SA's 30.7 at 1× — ABC wins the equal-time comparison; at equal budget SA ≥ ABC.
- ABC run-to-run stability: within-config std 1.05 spread units (CV 1.7%).

## 10. CELF breaks — the audit

Across all CELF runs: **50.2%** of lazy-bound re-checks violated the bound
(418/832). Decision-relevant audit (`results/val_celf_audit.csv`): CELF's picked set
differs from exact Greedy's in **16 of 17** (graph, k) configs; in **13 of 17** it
scores *strictly* below Greedy; only 1 of the differing cases is a near-tie.
Under submodularity this would be impossible — it is the empirical price of the lazy
bound on this objective.

## 11. Exact optima on small graphs (S2, held-out corrected)

Brute-force OPT, with the OPT set itself re-scored on held-out draws to avoid
best-of-many bias. ABC (proposed) held-out ratio: karate k=4 **1.003**, lesmis k=3
**1.000**, florentine k=4 **1.000**, davis k=3 **0.977**, star60/cycle60 **1.000**.
CELF+LS reaches OPT on karate (the 2-swap phase repairs the coalition 1-opt cannot).
On grid100 (a pathological plateau case) the whole family saturates at 81% of OPT at
equal budget; with 17× budget ABC reaches 97%.

## 12. What survives scrutiny (write-up core)

1. **The model is monotone and not submodular** — proved, with machine-checked
   witnesses and the Mossel–Roch sufficient-condition analysis.
2. **CELF breaks** — 50% lazy violations; worse than exact Greedy in 13/17 configs.
3. **Set-level search beats marginal search on sparse clustered graphs** — ABC vs
   Greedy +4.6 held-out on 16 independent WS instances (14/16, CI [+2.5, +6.9],
   p = 3×10⁻⁴, Bonferroni-safe), concentrated below Greedy's cascade threshold k*;
   ties Greedy on BA/powerlaw; beats CELF everywhere (16/16 on WS, 8/8 on BA/SBM).
4. **SA ≥ ABC**: on WS (CI [−2.8, −0.1]) and BA (p = 0.039); ABC is a valid,
   budget-efficient set-level searcher with a ≥CELF guarantee — not the best one.
5. **Signed extension**: generalises the model and exhibits non-monotonicity, but the
   search advantage over Greedy is a plateau effect, not a sign effect.

## 13. Reproducibility

- Modules: `majority_rule_model.py` (model, GPU backend, pools, baselines, greedy_phi,
  BFSBall/DenseCluster), `signed_rule_model.py`, `theory_checks.py`, `abc_celf.py`,
  `baselines.py` (SA, GRASP, sandwich-CELF), `run_experiments.py` (S1–S11, figures,
  summary), `run_validation.py` (OPT, leave-one-out, CELF audit, signed), `extend_ws.py`.
- Stages: `python run_experiments.py <s1|s2|s5|s5:ws|s8|s9|s10|s11|fig>`.
  All long studies append rows to their CSV after every cell and resume on re-run;
  nothing is lost to interruption.
- Key outputs: `results/s10_ws16_stats.json` (primary), `results/s10_multi_stats.json`
  (families), `results/summary.json` (S3 + signed + wall-clock + cascade success),
  `results/s1_structure.json` (theory), `figures/fig1..fig8`.
- GPU (RTX 3060) accelerates every cascade evaluation via a torch backend; CPU
  fallback is automatic (`backend="auto"`).

## 14. Suggested slide logic

1. Model + one figure (local activation function, fig6 left).
2. Theory: monotone (proof) / not submodular (K₁,₃ counterexample + scan table).
3. Why greedy/CELF fail: the plateau; CELF audit (16/17, 50%).
4. ABC design: canonical weaknesses → our operators (table §4) → final config from LOO.
5. Headline: WS forest plot (fig8) + family table with CIs; primary comparison stated.
6. Regime: gain vs k−k* window.
7. Budget/wall-clock: ABC ceiling at 1×, SA needs 4×; equal-time ABC wins.
8. Honest caveats: SA ≥ ABC; signed = plateau; δ-sweep confound; Φ negative.
9. Signed extension: model + non-monotonicity witness + sign control.
10. Conclusions §12.