# DECK HANDOFF — read me first

## What this is
A React slide deck (`deck/`, Vite + Recharts, 16 slides, 1280×720) presenting the project
"Influence Maximisation under the Majority-Rule Diffusion Model — monotone, not submodular;
an ABC optimiser; honest equal-budget/held-out evaluation". Base font size was increased
for readability; padding and gaps were tightened to keep content within the slide.

**How to run:** `cd deck` → `npm run dev` (or `npm run build` + `npm run preview`).
Slide N via `http://localhost:4173/#(N-1)`; arrows/space navigate.
**Screenshots:** `node screenshot.mjs` (uses system Edge, no download) → `deck/screens/slide-NN.png`.
The hash-routing bug that made all slides identical is FIXED — 16 distinct PNGs exist.
Re-take any screenshot you need and verify visually.

## The five claims on the slides (all held-out R=1000, equal budgets, graph = unit)
1. Model is monotone (proved; 5,800 pairs, 0 violations) and NOT submodular (K₁,₃ counterexample
   3.0 vs 1.0; scans 23–76% non-deg violations vs 0.0% LT control; γ ∈ [0.29, 0.85]).
2. CELF breaks: 50.2% lazy-bound violations (418/832); differs from Greedy 16/17 configs,
   strictly worse 13/17; on ba500 k=30: CELF 131.9 vs Degree 473.7 vs Greedy 491.5 vs ABC 493.8.
3. Primary (pre-specified): ABC − Greedy on 16 fresh WS(1000) instances = +4.57 held-out,
   CI [+2.46, +6.91], 14/16 wins, p = 3.1e-4 (Bonferroni-safe). SA − Greedy = +5.99 (15/16).
   BA(8): wash +0.26; SBM(8): suggestive +0.58. Constructive BFS/cluster baselines far behind.
4. SA ≥ ABC (WS CI [−2.84, −0.12]; BA p = 0.039) — stated plainly. ABC: budget-efficient
   (ceiling at 1× vs SA 4×), ≥CELF guarantee, CV 1.7%, cheapest set-level searcher (4.5 s).
5. Signed: net-support model, non-monotone (hub-blocker + drop-0.26 witness). Sign-ablation
   control: advantage persists on all-positive twin (sbm120 k=15: +181% vs +95%) → plateau,
   not signs. Negative results: Φ fitness, Greedy+Φ, refinement reserve, Metropolis default,
   δ-sweep correlation (confounded: partial rho kills it, p ≥ 0.31).

## Slide-by-slide checklist (verify: layout, no overflow, charts render, numbers match)
1. Title — gradient, three chips, 5 claims teaser.
2. Model — formula panel, 4 variant chips, estimator note, signed note, degree-need chips.
3. Structure — monotonicity proof box; K₁,₃ counterexample table; dual-line δ-sweep chart
   (red K₁,₃, blue K₁,₉; positive through δ=0.4, negative at 0.45–0.5).
4. Scan — 12-row table (LT rows highlighted; γ column); KPIs 0.29–0.85 / 0.96–0.99 / 0.0%;
   Mossel–Roch panel; Lu et al. corollary panel.
5. CELF — 3 red KPIs; bar chart CELF 131.9 / Degree 473.7 / Greedy 491.5 / ABC 493.8
   (red/grey/orange/teal); mechanism + sandwich panels.
6. ABC — W1–W9 table (gbest pos, reserve neg); LOO horizontal bar chart (teal −gbest,
   red −reserve, grey rest); final-config note.
7. Protocol — 3 panels (equal budget / held-out / graph unit) + 2 panels (primary/Bonferroni,
   baselines chips).
8. Primary — custom SVG forest (16 graphs, teal ABC−Greedy, blue ABC−SA, green CI band);
   stats table (4 rows with CIs); cascade-fraction bar chart (33/43/51/54%).
9. Families — BA panel (3 KPIs, wash text), SBM panel (suggestive), WS means bar chart
   (Greedy 20.0, BFS 13.6, Dense 12.9, ABC 24.6, SA 26.0).
10. Regime — bar chart gain vs k−k* (+8.42 at −10 … +3.13 at +15, all labeled); KPIs
    +64.6% / +23.0% / ≈0; percolation panel ("no random k-set cascades, k ≤ 20").
11. Budget — line chart (teal ABC 36.1→37.45 flat, blue SA 35.2→37.6, red dashed Greedy 27.5,
    dotted ba500 ≈94); ceiling panel; wall-clock horizontal bars (CELF 0.18 … SA 10.71).
12. δ-sweep — ws1000 advantage line (+75/+27/+35/+24/+53 → ≈0); ba500 dual line (orange vs
    CELF +54→+0.4, teal vs Greedy ≈0); amber confound warning.
13. Signed — witness formula (σ({1,2,3,4})=6 > σ({1,2,3,4,5})=5); KPIs 17.6/17.7/15.4/14.6;
    sign-control grouped bars (sbm120 19.9→38.9 vs twin 19.8→55.6; ws300 21.0→26.9 vs 36.8→40.7).
14. Implementation story — original pipeline / shortcomings / fixes / failures / final
    submission package.
15. Conclusions — 5 "supports" + 3 "don't claim" (SA ≥ ABC; δ-sweep confound; signed=plateau).
16. Future scope — current limitations, next steps, final framing.

## Key files
- deck/src/slides.jsx (all slide components), deck/src/ui.jsx (KPI/Table/Forest/CIbadge),
  deck/src/data.js (every number used), deck/src/deck.css (design system).
- Project results: ../results/s10_ws16_stats.json, s10_multi_stats.json, summary.json,
  val_celf_audit.csv, val_loo.csv, s9_budget.csv, s5_delta_*.csv, s11_signcontrol.csv,
  s2_exact.csv, s8_signed.csv.
- Full write-up: ../MAJORITY_RULE_ABC_REPORT.md; quick summary: ../FINAL_FINDINGS.md.
- Figures (matplotlib, for reference): ../figures/fig1…fig8.

## Known non-issues
- Vite "chunk > 500 kB" warning is benign. The Playwright DEP0190 warning is benign.
- Screenshots use a 1312×806 viewport; the .slide is exactly 1280×720.