// All numbers are held-out (R=1000), equal-budget results from
// results/*.csv of the project. Primary comparison: ABC vs Greedy on WS.

export const C = {
  teal: '#16a085', tealInk: '#0e7a65',
  red: '#c0392b', blue: '#2980b9', orange: '#e67e22',
  gold: '#f1c40f', grey: '#8a93a3', ink: '#182130',
}

export const sweepK13 = [
  { d: 0.0, k13: 2.0, k19: 5.0 },
  { d: 0.05, k13: 2.0, k19: null },
  { d: 0.1, k13: 2.0, k19: null },
  { d: 0.2, k13: 1.33, k19: 1.05 },
  { d: 0.3, k13: 0.25, k19: 0.54 },
  { d: 0.35, k13: null, k19: 0.4 },
  { d: 0.4, k13: null, k19: 0.16 },
  { d: 0.45, k13: null, k19: -0.41 },
  { d: 0.5, k13: -0.62, k19: -0.88 },
]

export const structureTable = [
  { g: 'karate', v: 'LT (0.50)', mono: 0, viol: '0.0%', nd: '0.0%', gam: 0.992, ctl: true },
  { g: 'karate', v: 'majority (0.30)', mono: 0, viol: '57.6%', nd: '66.3%', gam: 0.511 },
  { g: 'karate', v: 'majority (0.15)', mono: 0, viol: '51.2%', nd: '68.7%', gam: 0.345 },
  { g: 'karate', v: 'majority (det.)', mono: 0, viol: '25.8%', nd: '52.6%', gam: 0.292 },
  { g: 'lesmis', v: 'LT (0.50)', mono: 0, viol: '0.0%', nd: '0.0%', gam: 0.967, ctl: true },
  { g: 'lesmis', v: 'majority (0.30)', mono: 0, viol: '33.6%', nd: '64.4%', gam: 0.462 },
  { g: 'lesmis', v: 'majority (0.15)', mono: 0, viol: '23.1%', nd: '60.4%', gam: 0.653 },
  { g: 'lesmis', v: 'majority (det.)', mono: 0, viol: '6.7%', nd: '47.0%', gam: 0.333 },
  { g: 'ba500', v: 'LT (0.50)', mono: 0, viol: '0.0%', nd: '0.0%', gam: 0.963, ctl: true },
  { g: 'ba500', v: 'majority (0.30)', mono: 0, viol: '45.2%', nd: '59.1%', gam: 0.845 },
  { g: 'ba500', v: 'majority (0.15)', mono: 0, viol: '52.0%', nd: '75.7%', gam: 0.693 },
  { g: 'ba500', v: 'majority (det.)', mono: 0, viol: '33.3%', nd: '70.4%', gam: 0.667 },
]

export const exactTable = [
  { g: 'karate (4)', deg: 0.955, gr: 0.977, celf: 0.917, ls: 1.0, abc: 1.003 },
  { g: 'lesmis (3)', deg: 0.744, gr: 1.0, celf: 0.872, ls: 0.884, abc: 1.0 },
  { g: 'florentine (4)', deg: 0.937, gr: 1.0, celf: 0.833, ls: 0.937, abc: 1.0 },
  { g: 'davis (3)', deg: 0.95, gr: 1.0, celf: 0.741, ls: 0.76, abc: 0.977 },
  { g: 'star60 (3)', deg: 1.0, gr: 1.0, celf: 1.0, ls: 1.0, abc: 1.0 },
  { g: 'cycle60 (3)', deg: 0.535, gr: 1.0, celf: 1.0, ls: 1.0, abc: 1.0 },
]

export const celfAudit = [
  { kpi: '50.2%', label: 'lazy-bound violations over all CELF runs (418 / 832 re-checks)' },
  { kpi: '16 / 17', label: 'configs where CELF’s picked set differs from exact Greedy’s' },
  { kpi: '13 / 17', label: 'configs where CELF scores strictly below Greedy (held-out)' },
  { kpi: '131.9', label: 'CELF held-out on ba500 k=30 — vs Greedy 491.5, Degree 473.7, ABC 493.8' },
]

export const looTable = [
  { op: '− gbest guide', d: -0.132, p: '0.0006', sig: true, note: 'the one operator that matters' },
  { op: '− refinement reserve', d: +0.116, p: '0.012', sig: true, note: 'removing it helps → refinement dropped' },
  { op: '− tournament', d: +0.020, p: '0.40', sig: false, note: '' },
  { op: '− directed scouts', d: -0.013, p: '0.68', sig: false, note: '' },
  { op: '− Φ tie-break', d: +0.004, p: '0.75', sig: false, note: 'neutral → kept (WS plateaus)' },
  { op: '− dedupe / −accept-equal / −explore / −2-swap', d: '−0.02…+0.04', p: 'ns', sig: false, note: 'all < 1% effects' },
]

export const ws16 = {
  primary: { mean: 4.57, ci: [2.46, 6.91], wins: '14 / 16', p: '3.1e-4', bonf: true },
  saVgreedy: { mean: 5.99, ci: [4.01, 7.85], wins: '15 / 16', p: '1.5e-4' },
  abcVSsa: { mean: -1.41, ci: [-2.84, -0.12], wins: '6 / 16', p: '0.093' },
  abcVCelf: { mean: 2.69, ci: [1.83, 3.89], wins: '16 / 16', p: '3.1e-5' },
  means: { SA: 26.0, ABC: 24.6, CELF_LS: 24.1, CELF: 21.9, Greedy: 20.0, BFS: 13.6, Dense: 12.9 },
  forest: [
    { g: 'ws_6_9', abc: -0.7, sa: 0.78 },
    { g: 'ws_6_6', abc: -0.84, sa: -0.96 },
    { g: 'ws_6_15', abc: 0.03, sa: -3.14 },
    { g: 'ws_6_12', abc: 1.0, sa: 0.7 },
    { g: 'ws_8_13', abc: 1.53, sa: -5.97 },
    { g: 'ws_6_3', abc: 2.05, sa: -3.56 },
    { g: 'ws_6_0', abc: 2.11, sa: -4.15 },
    { g: 'ws_8_4', abc: 2.6, sa: -7.0 },
    { g: 'ws_8_7', abc: 4.42, sa: 0.72 },
    { g: 'ws_8_1', abc: 4.46, sa: -0.59 },
    { g: 'ws_8_10', abc: 5.99, sa: -0.78 },
    { g: 'ws_10_2', abc: 6.78, sa: -2.48 },
    { g: 'ws_10_8', abc: 6.42, sa: -2.31 },
    { g: 'ws_10_5', abc: 11.69, sa: 1.76 },
    { g: 'ws_10_11', abc: 11.81, sa: 1.93 },
    { g: 'ws_10_14', abc: 13.83, sa: 2.41 },
  ],
  cascade: { ABC: 0.542, SA: 0.509, CELF: 0.428, Greedy: 0.33 },
}

export const famStats = {
  BA: {
    abcGreedy: { mean: 0.26, ci: [-0.42, 1.06], wins: '5 / 8', p: '0.64', label: 'wash' },
    abcSa: { mean: -0.49, ci: [-0.79, -0.18], wins: '2 / 8', p: '0.039', label: 'SA ≥ ABC' },
    abcCelf: { mean: 12.4, ci: [9.3, 16.0], wins: '8 / 8', p: '0.0078', label: '≫ CELF' },
    means: { SA: 40.6, ABC: 40.1, Greedy: 39.8, CELF: 27.7, BFS: 22.9 },
  },
  SBM: {
    abcGreedy: { mean: 0.58, ci: [0.07, 1.14], wins: '6 / 8', p: '0.078', label: 'suggestive' },
    abcSa: { mean: -0.05, ci: [-0.34, 0.20], wins: '4 / 8', p: '0.84', label: 'tie' },
    abcCelf: { mean: 1.57, ci: [1.02, 2.15], wins: '8 / 8', p: '0.0078', label: '> CELF' },
    means: { SA: 13.7, ABC: 13.6, Greedy: 13.1, CELF: 12.1, BFS: 11.7 },
  },
}

export const regime = [
  { dk: '-10', gain: 8.42, n: 2 },
  { dk: '-5', gain: 2.84, n: 1 },
  { dk: '0 (k*)', gain: 5.53, n: 8 },
  { dk: '+5', gain: 2.16, n: 5 },
  { dk: '+10', gain: 4.53, n: 1 },
  { dk: '+15', gain: 3.13, n: 5 },
]
export const regimeRel = { below: 64.6, above: 23.0 }

export const budgetData = [
  { mult: '0.5×', wsABC: 36.13, wsSA: 35.17, wsG: 27.49, baABC: 93.3, baSA: 92.34 },
  { mult: '1×', wsABC: 37.45, wsSA: 30.66, wsG: 27.49, baABC: 93.3, baSA: 92.85 },
  { mult: '4×', wsABC: 37.45, wsSA: 37.59, wsG: 27.49, baABC: 93.95, baSA: 93.07 },
  { mult: '8×', wsABC: 37.45, wsSA: 35.59, wsG: 27.49, baABC: 93.95, baSA: 93.95 },
]

export const wallClock = [
  { a: 'CELF', s: 0.18 },
  { a: 'CELF+LS', s: 0.74 },
  { a: 'ABC', s: 4.5 },
  { a: 'Greedy', s: 5.96 },
  { a: 'GRASP', s: 6.04 },
  { a: 'SA', s: 10.71 },
]

export const deltaBa = [
  { d: 0.0, vsC: 54.4, vsG: 2.9, gam: 0.56 },
  { d: 0.05, vsC: 41.1, vsG: 0.6, gam: 0.52 },
  { d: 0.1, vsC: 38.7, vsG: -1.7, gam: 0.52 },
  { d: 0.15, vsC: 39.8, vsG: -0.5, gam: 0.53 },
  { d: 0.2, vsC: 10.7, vsG: -1.9, gam: 0.68 },
  { d: 0.3, vsC: 2.8, vsG: -0.9, gam: 0.82 },
  { d: 0.4, vsC: 0.6, vsG: -0.5, gam: 0.8 },
  { d: 0.5, vsC: 0.4, vsG: 0.4, gam: 0.95 },
]
export const deltaWs = [
  { d: 0.0, vsG: 75.0 },
  { d: 0.05, vsG: 27.3 },
  { d: 0.1, vsG: 34.8 },
  { d: 0.15, vsG: 24.4 },
  { d: 0.2, vsG: 53.2 },
  { d: 0.3, vsG: -1.2 },
  { d: 0.4, vsG: -3.4 },
  { d: 0.5, vsG: -0.1 },
]

export const signControl = [
  { g: 'sbm120 · k=15 · signed', greedy: 19.9, abc: 38.9 },
  { g: 'sbm120 · k=15 · positive twin', greedy: 19.8, abc: 55.6 },
  { g: 'ws300 · k=15 · signed', greedy: 21.0, abc: 26.9 },
  { g: 'ws300 · k=15 · positive twin', greedy: 36.8, abc: 40.7 },
]

export const abFormulas = {
  mono: 'S ⊆ T  ⟹  A_t(S) ⊆ A_t(T)  ⟹  σ(S) ≤ σ(T)',
  sub: 'σ(S+v) − σ(S)  ≥  σ(T+v) − σ(T)   for all  S ⊆ T, v ∉ T',
}