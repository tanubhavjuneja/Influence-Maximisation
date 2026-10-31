const PptxGenJS = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Influence Maximisation under the Majority-Rule Diffusion Model';
pres.author = 'Majority-Rule ABC Study';
pres.subject = 'Editable deck: monotone, not submodular; ABC; equal-budget held-out evaluation';

const PW = 13.333;
const PH = 7.5;

const C = {
  navy: '182130',
  navy2: '1B3A5C',
  teal: '16A085',
  teal2: '0E7A65',
  red: 'C0392B',
  red2: 'F1948A',
  blue: '2980B9',
  orange: 'E67E22',
  grey: '5D6D7E',
  lgrey: 'F4F6FB',
  ice: 'D7E0F5',
  white: 'FFFFFF',
  black: '1B1F3B',
  gold: 'F1C40F',
};

function sizeOf(p) {
  const buf = fs.readFileSync(p);
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}
function fitImage(p, maxW, maxH) {
  const { width, height } = sizeOf(p);
  const ar = width / height;
  let w = maxW, h = w / ar;
  if (h > maxH) { h = maxH; w = h * ar; }
  return { w, h };
}

let pageNum = 0;
function footer(slide, dark = false) {
  pageNum++;
  slide.addText(`${pageNum}`, {
    x: PW - 0.7, y: PH - 0.45, w: 0.5, h: 0.3,
    fontSize: 10, color: dark ? 'AAB4D4' : '9AA3B5', align: 'right',
  });
  slide.addText('Majority-Rule Diffusion — ABC Study', {
    x: 0.5, y: PH - 0.45, w: 6, h: 0.3,
    fontSize: 10, color: dark ? 'AAB4D4' : '9AA3B5',
  });
}

function sectionTag(slide, text, color = C.teal) {
  slide.addText(text.toUpperCase(), {
    x: 0.55, y: 0.4, w: 8, h: 0.32,
    fontSize: 12, bold: true, color,
  });
}

function slideTitle(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.5, y: opts.y ?? 0.75, w: opts.w ?? 12.3, h: opts.h ?? 0.75,
    fontSize: opts.size ?? 32, bold: true, color: opts.color ?? C.navy,
    valign: 'top',
  });
}

function noteBox(slide, text, x, y, w, h, opts = {}) {
  slide.addShape('roundRect', {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: opts.fill || C.lgrey },
    line: opts.line ? { color: opts.line, width: 1 } : { type: 'none' },
  });
  slide.addText(text, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: h - 0.2,
    fontSize: opts.size || 13, color: opts.color || C.black,
    valign: 'middle',
  });
}

const root = path.join(__dirname, '..');
function img(slide, rel, maxW, maxH, cx, cy) {
  const p = path.join(root, rel);
  if (!fs.existsSync(p)) return null;
  const { w, h } = fitImage(p, maxW, maxH);
  const x = cx == null ? (PW - w) / 2 : cx;
  const y = cy == null ? (PH - h) / 2 : cy;
  slide.addImage({ path: p, x, y, w, h });
  return { w, h };
}

// ============================================================ 1. TITLE
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape('rect', { x: 0, y: 0, w: PW, h: PH, fill: { color: C.navy }, line: { type: 'none' } });
  s.addShape('ellipse', { x: 10.7, y: -1.4, w: 5.5, h: 5.5, fill: { color: '141B47' }, line: { type: 'none' } });
  s.addShape('ellipse', { x: 12.2, y: 3.6, w: 3.6, h: 3.6, fill: { color: '141B47' }, line: { type: 'none' } });
  s.addShape('ellipse', { x: 12.6, y: 3.3, w: 1.1, h: 1.1, fill: { color: C.teal }, line: { type: 'none' } });

  s.addText('TECHNICAL ANALYSIS', {
    x: 0.9, y: 1.45, w: 8, h: 0.35,
    fontSize: 13, bold: true, color: C.teal, charSpacing: 2,
  });
  s.addText('Influence Maximisation under the\nMajority Rule Diffusion Model', {
    x: 0.9, y: 1.9, w: 10.5, h: 1.9,
    fontSize: 40, bold: true, color: C.white, lineSpacing: 42,
  });
  s.addText('The model is monotone, not submodular — and that is why set-level search beats greedy\nAn Artificial Bee Colony optimiser with equal-budget, held-out evaluation', {
    x: 0.9, y: 4.0, w: 10.5, h: 0.9,
    fontSize: 17, italic: true, color: C.ice, lineSpacing: 22,
  });
  s.addShape('line', { x: 0.9, y: 5.25, w: 2.4, h: 0, line: { color: C.teal, width: 2.5 } });
  s.addText('September 2026', {
    x: 0.9, y: 5.45, w: 6, h: 0.35,
    fontSize: 13, color: 'AAB4D4',
  });
}

// ============================================================ 2. HEADLINE FINDING
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  sectionTag(s, 'The correction', C.teal);
  s.addText('The brief asked to prove the model is\n"submodular and monotone."', {
    x: 0.7, y: 0.9, w: 11.8, h: 1.1,
    fontSize: 24, bold: true, color: C.white, lineSpacing: 30,
  });

  const cardW = 5.7, cardH = 3.6, gap = 0.5, y = 2.7;
  s.addShape('roundRect', {
    x: 0.7, y, w: cardW, h: cardH, rectRadius: 0.1,
    fill: { color: '1B3A5C' }, line: { color: C.teal, width: 1.5 },
  });
  s.addText('MONOTONE', { x: 1.0, y: y + 0.3, w: cardW - 0.6, h: 0.45, fontSize: 14, bold: true, color: C.teal, charSpacing: 2 });
  s.addText('TRUE', { x: 1.0, y: y + 0.75, w: cardW - 0.6, h: 1.0, fontSize: 52, bold: true, color: C.white });
  s.addText('Proved by coupling; verified with zero violations across 5,800 sampled pairs in every (graph × variant) configuration.',
    { x: 1.0, y: y + 1.95, w: cardW - 0.6, h: 1.4, fontSize: 14, color: C.ice, lineSpacing: 18 });

  const x2 = 0.7 + cardW + gap;
  s.addShape('roundRect', {
    x: x2, y, w: cardW, h: cardH, rectRadius: 0.1,
    fill: { color: '4A2020' }, line: { color: C.red, width: 1.5 },
  });
  s.addText('SUBMODULAR', { x: x2 + 0.3, y: y + 0.3, w: cardW - 0.6, h: 0.45, fontSize: 14, bold: true, color: C.red2, charSpacing: 2 });
  s.addText('FALSE', { x: x2 + 0.3, y: y + 0.75, w: cardW - 0.6, h: 1.0, fontSize: 52, bold: true, color: C.white });
  s.addText('Disproved by a K1,3 counterexample and exhaustive scans: 23–76% of non-degenerate triples violate diminishing returns.',
    { x: x2 + 0.3, y: y + 1.95, w: cardW - 0.6, h: 1.4, fontSize: 14, color: 'F5D5D0', lineSpacing: 18 });
  footer(s, true);
}

// ============================================================ 3. THE MODEL
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'The model');
  slideTitle(s, 'Majority-rule threshold diffusion');

  s.addText('A node activates once the count of active neighbours exceeds a threshold fraction of its degree:', {
    x: 0.55, y: 1.7, w: 11.5, h: 0.45, fontSize: 16, color: C.black,
  });
  s.addShape('roundRect', {
    x: 0.55, y: 2.2, w: 8.0, h: 1.0, rectRadius: 0.08,
    fill: { color: C.lgrey }, line: { type: 'none' },
  });
  s.addText('|{active neighbours of v}|   >   θv · deg(v)', {
    x: 0.55, y: 2.2, w: 8.0, h: 1.0, align: 'center', valign: 'middle',
    fontSize: 22, italic: true, bold: true, color: C.navy,
  });

  const rows = [
    ['Deterministic majority (dmr)', 'θv = 1/2', C.navy],
    ['Stochastic majority (smr)', 'θv ~ U[1/2−δ, 1/2+δ]', C.grey],
    ['Linear Threshold (control)', 'θv ~ U[0,1]  —  submodular', C.teal],
  ];
  let y = 3.6;
  rows.forEach((r) => {
    s.addShape('rect', { x: 0.55, y, w: 0.12, h: 0.7, fill: { color: r[2] }, line: { type: 'none' } });
    s.addText(r[0], { x: 0.8, y: y - 0.02, w: 3.8, h: 0.4, fontSize: 15, bold: true, color: C.navy });
    s.addText(r[1], { x: 0.8, y: y + 0.35, w: 7.2, h: 0.4, fontSize: 14, color: C.grey });
    y += 0.9;
  });

  s.addShape('roundRect', {
    x: 9.1, y: 1.7, w: 3.75, h: 4.8, rectRadius: 0.1,
    fill: { color: C.navy }, line: { type: 'none' },
  });
  s.addText('Key property', { x: 9.4, y: 2.0, w: 3.2, h: 0.4, fontSize: 14, bold: true, color: C.teal });
  s.addText('A neighbour is either worthless or decisive. Until a coalition reaches the majority threshold, singleton marginal gains are near zero — so greedy methods that look one node at a time are blind to the winning coalition.', {
    x: 9.4, y: 2.5, w: 3.2, h: 2.2, fontSize: 14, color: C.ice, lineSpacing: 18,
  });
  footer(s);
}

// ============================================================ 4. PROOF OF MONOTONICITY
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Proof — Part 1');
  slideTitle(s, 'Monotonicity: σ(S) ≤ σ(T) for S ⊆ T');

  const steps = [
    ['Fix', 'a threshold realisation θ and couple cascades started from S and T.'],
    ['Induct', 'on time t: assume At(S) ⊆ At(T). True at t = 0 since S ⊆ T.'],
    ['Step', 'if v activates under S at t+1, |N(v)∩At(S)| > θv deg(v); because At(S) ⊆ At(T), v also activates under T.'],
    ['Conclude', 'progressivity preserves the invariant forever; expectation keeps σ(S) ≤ σ(T).'],
  ];
  const w = 2.92, gap = 0.14, x0 = 0.55, y0 = 2.0, h = 3.4;
  steps.forEach((st, i) => {
    const x = x0 + i * (w + gap);
    s.addShape('roundRect', {
      x, y: y0, w, h, rectRadius: 0.08,
      fill: { color: i % 2 ? C.lgrey : 'EDF1FB' }, line: { type: 'none' },
    });
    s.addShape('ellipse', { x: x + 0.25, y: y0 + 0.3, w: 0.55, h: 0.55, fill: { color: C.teal }, line: { type: 'none' } });
    s.addText(String(i + 1), { x: x + 0.25, y: y0 + 0.3, w: 0.55, h: 0.55, align: 'center', valign: 'middle', fontSize: 18, bold: true, color: C.white });
    s.addText(st[0], { x: x + 0.25, y: y0 + 1.05, w: w - 0.5, h: 0.4, fontSize: 17, bold: true, color: C.navy });
    s.addText(st[1], { x: x + 0.25, y: y0 + 1.55, w: w - 0.5, h: 1.8, fontSize: 13.5, color: C.black, lineSpacing: 17 });
  });

  s.addShape('roundRect', {
    x: 0.55, y: 5.75, w: 12.2, h: 0.85, rectRadius: 0.08,
    fill: { color: C.teal }, line: { type: 'none' },
  });
  s.addText('Empirically verified: 0 violations across 5,800 sampled (S ⊆ T) pairs in 12 (graph × variant) configurations.',
    { x: 0.85, y: 5.75, w: 11.6, h: 0.85, valign: 'middle', fontSize: 15, bold: true, color: C.white });
  footer(s);
}

// ============================================================ 5. DISPROOF
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Proof — Part 2', C.red);
  slideTitle(s, 'Disproof: the swing-voter counterexample');

  s.addText('Star graph K1,3: centre w needs 2 of its 3 neighbours active.', {
    x: 0.55, y: 1.7, w: 6.4, h: 0.5, fontSize: 15, color: C.grey, lineSpacing: 18,
  });
  const cx = 3.0, cy = 4.3, r = 1.5;
  s.addShape('line', { x: cx, y: cy, w: 0, h: -r, line: { color: '8FA9DE', width: 2 } });
  s.addShape('line', { x: cx, y: cy, w: -r * 0.87, h: r * 0.5, line: { color: '8FA9DE', width: 2 } });
  s.addShape('line', { x: cx, y: cy, w: r * 0.87, h: r * 0.5, line: { color: '8FA9DE', width: 2 } });
  s.addShape('ellipse', { x: cx - 0.32, y: cy - 0.32, w: 0.64, h: 0.64, fill: { color: C.navy }, line: { color: C.white, width: 2 } });
  s.addText('w', { x: cx - 0.32, y: cy - 0.32, w: 0.64, h: 0.64, align: 'center', valign: 'middle', fontSize: 14, bold: true, color: C.white });
  const leaves = [[cx, cy - r, 'x'], [cx - r * 0.87, cy + r * 0.5, 'z'], [cx + r * 0.87, cy + r * 0.5, 'y']];
  leaves.forEach(([lx, ly, lbl]) => {
    const col = lbl === 'y' ? C.teal : '8FA9DE';
    s.addShape('ellipse', { x: lx - 0.28, y: ly - 0.28, w: 0.56, h: 0.56, fill: { color: col }, line: { color: C.white, width: 2 } });
    s.addText(lbl, { x: lx - 0.28, y: ly - 0.28, w: 0.56, h: 0.56, align: 'center', valign: 'middle', fontSize: 13, bold: true, color: C.white });
  });
  s.addText('y = the swing voter under test', { x: 0.55, y: 6.1, w: 5.5, h: 0.35, fontSize: 12, italic: true, color: C.teal });

  const rx = 7.1, rw = 5.6;
  s.addShape('roundRect', { x: rx, y: 1.85, w: rw + 0.3, h: 2.1, rectRadius: 0.08, fill: { color: C.lgrey }, line: { type: 'none' } });
  s.addText('S = ∅  →  S ∪ {y}', { x: rx + 0.3, y: 2.05, w: rw, h: 0.4, fontSize: 15, bold: true, color: C.navy });
  s.addText('σ({y}) − σ(∅) = 1.0', { x: rx + 0.3, y: 2.5, w: rw, h: 0.45, fontSize: 22, color: C.black });
  s.addText('y alone cannot flip w', { x: rx + 0.3, y: 3.05, w: rw, h: 0.35, fontSize: 12, italic: true, color: C.grey });

  s.addShape('roundRect', { x: rx, y: 4.05, w: rw + 0.3, h: 2.1, rectRadius: 0.08, fill: { color: 'FBEAE8' }, line: { color: C.red, width: 1 } });
  s.addText('T = {x}  →  T ∪ {y}', { x: rx + 0.3, y: 4.25, w: rw, h: 0.4, fontSize: 15, bold: true, color: C.navy });
  s.addText('σ({x,y}) − σ({x}) = 3.0', { x: rx + 0.3, y: 4.7, w: rw, h: 0.45, fontSize: 22, bold: true, color: C.red });
  s.addText('x and y together flip w — gain triples for the larger set', { x: rx + 0.3, y: 5.25, w: rw, h: 0.45, fontSize: 12, italic: true, color: C.red });
  footer(s);
}

// ============================================================ 6. WHY IT FAILS
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Why it fails');
  slideTitle(s, 'Threshold CDF concavity breaks submodularity');
  s.addText('Mossel & Roch (2007): σ is submodular only if every local activation function is concave in the active-neighbour count. A majority requirement makes it S-shaped — convex on the lower stretch — breaking that condition by construction.', {
    x: 0.55, y: 1.65, w: 12.2, h: 0.85, fontSize: 15, color: C.black, lineSpacing: 19,
  });
  img(s, 'figures/fig6_structure.png', 11.8, 4.4, 0.75, 2.75);
  footer(s);
}

// ============================================================ 7. EXHAUSTIVE VERIFICATION
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Verification');
  slideTitle(s, 'Exhaustive verification across graphs');

  const rows = [
    [{ text: 'Graph', options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: 'Variant', options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: 'Monotone', options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: 'Non-deg. violations', options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: 'γmin', options: { bold: true, color: C.white, fill: { color: C.navy } } }],
    ['karate', 'LT (control)', '✓', '0.0%', '0.992'],
    ['karate', 'Majority (δ=0.15)', '✓', '68.7%', '0.345'],
    ['lesmis', 'LT (control)', '✓', '0.0%', '0.967'],
    ['lesmis', 'Majority (δ=0.15)', '✓', '60.4%', '0.653'],
    ['ba500', 'LT (control)', '✓', '0.0%', '0.963'],
    ['ba500', 'Majority (δ=0.15)', '✓', '75.7%', '0.693'],
  ];
  s.addTable(rows, {
    x: 0.6, y: 2.0, w: 12.0, h: 3.0,
    fontFace: 'Calibri', fontSize: 14,
    colW: [2.2, 3.5, 1.8, 2.6, 1.9],
    color: C.black,
    valign: 'middle',
    align: 'center',
    border: { pt: 0.75, color: 'DFE3EE' },
    fill: { color: C.lgrey },
  });
  noteBox(s, 'Green rows = Linear Threshold control. Its clean 0.0% violation rate calibrates the test, confirming that majority-rule violations are real, not noise.', 0.6, 5.35, 12.0, 0.65, { fill: 'E6F5F1' });
  footer(s);
}

// ============================================================ 8. CELF BREAKS
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Consequence', C.red);
  slideTitle(s, 'CELF\'s lazy bound is wrong half the time');

  const kpis = [
    ['50.2%', 'lazy-bound violations over all CELF runs\n418 of 832 re-checks'],
    ['16 / 17', 'configs where CELF differs from exact Greedy'],
    ['13 / 17', 'configs where CELF is strictly worse'],
  ];
  kpis.forEach((k, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape('roundRect', { x, y: 1.85, w: 3.9, h: 1.55, rectRadius: 0.08, fill: { color: 'FBEAE8' }, line: { type: 'none' } });
    s.addText(k[0], { x: x + 0.2, y: 2.0, w: 3.5, h: 0.75, fontSize: 30, bold: true, color: C.red });
    s.addText(k[1], { x: x + 0.2, y: 2.75, w: 3.5, h: 0.6, fontSize: 12, color: C.black, lineSpacing: 14 });
  });

  s.addText('The collapse on hub graphs (ba500, k = 30, held-out)', {
    x: 0.6, y: 3.7, w: 12.0, h: 0.35, fontSize: 15, bold: true, color: C.navy,
  });
  s.addChart('bar', [
    { name: 'held-out spread', labels: ['CELF', 'Degree', 'Greedy', 'ABC'], values: [131.9, 473.7, 491.5, 493.8] },
  ], {
    x: 0.6, y: 4.05, w: 7.2, h: 2.6,
    chartColors: [C.red, C.grey, C.orange, C.teal],
    showValue: true,
    dataLabelFontSize: 10,
    dataLabelColor: C.black,
  });
  noteBox(s, 'CELF loses to a trivial degree ranking. The +273% gap is CELF\'s collapse, not ABC\'s miracle — gains against CELF should be read this way.', 8.0, 4.05, 4.8, 2.6, { fill: C.lgrey });
  footer(s);
}

// ============================================================ 9. ABC OPTIMISER
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'The method', C.teal);
  slideTitle(s, 'ABC: upgrades chosen by a leave-one-out ablation');

  const rows = [
    ['Weak exploitation', 'gbest-guided partner (p = 0.6)', 'only operator that matters'],
    ['Roulette convergence', 'tournament selection (size 4)', 'kept'],
    ['Blind scouts', 'directed scouts near incumbent', 'kept'],
    ['Fixed perturbation', 'n_mod swaps + mutation', 'kept'],
    ['Duplicate sources', 'deduplication', 'kept'],
    ['Plateau rigidity', 'Φ tie-break + accept-equal + random moves', 'neutral'],
    ['Refinement budget', 'removed — it hurt (LOO)', 'REMOVED'],
    ['1-opt can\'t repair coalitions', '2-swap + cascade-aware candidates', 'in CELF+LS'],
  ];
  s.addTable(rows.map((r, i) => i === 0 ? r.map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy } } })) : r), {
    x: 0.55, y: 1.7, w: 6.8, h: 4.9,
    fontFace: 'Calibri', fontSize: 13,
    colW: [2.6, 3.2, 1.0],
    color: C.black,
    valign: 'middle',
    border: { pt: 0.75, color: 'DFE3EE' },
    fill: { color: C.lgrey },
  });

  s.addText('Leave-one-out ablation (Δ held-out)', { x: 7.6, y: 1.7, w: 5.0, h: 0.35, fontSize: 14, bold: true, color: C.navy });
  s.addChart('bar', [
    { name: 'Δ mean held-out', labels: ['−gbest', '−reserve', '−tournament', '−directed', '−Φ', '−dedupe/−explore'], values: [-0.132, 0.116, 0.020, -0.013, 0.004, 0.015] },
  ], {
    x: 7.6, y: 2.05, w: 5.2, h: 3.2,
    chartColors: [C.teal],
    barDir: 'bar',
    showValue: true,
    dataLabelFontSize: 10,
    dataLabelColor: C.black,
    barGapWidthPct: 40,
  });
  noteBox(s, 'Final method = CELF-seeded, gbest-guided population search with no refinement phase. Elitism guarantees it is never worse than CELF.', 7.6, 5.45, 5.2, 1.15, { fill: 'E6F5F1' });
  footer(s);
}

// ============================================================ 10. EVALUATION PROTOCOL
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Methodology');
  slideTitle(s, 'Equal budgets, held-out scoring, graph = unit');

  const items = [
    ['01', 'Equal budget', 'Every budgeted solver is capped at Greedy\'s cost: k × |pool| fresh cascade evaluations. Audit: 0 budget violations across 532 runs.'],
    ['02', 'Held-out scoring', 'Final seed sets are re-scored on 1,000 independent threshold draws. ABC in-sample − held-out gap: +0.13 (worst +4.4) — not an overfitting artefact.'],
    ['03', 'Graph = unit', 'Spreads are averaged over k within each graph before testing. Wilcoxon signed-rank + cluster-bootstrap CIs treat the graph, not the (graph,k) row, as the statistical unit.'],
    ['04', 'Pre-specified primary', 'One declared primary comparison: ABC vs Greedy on 16 WS graphs. Nine family tests → Bonferroni threshold 0.0056.'],
    ['05', 'Baselines isolate the claim', 'Zero-MC: Degree, PageRank, k-Core. Marginal: Greedy, CELF, CELF+LS. Set-level: ABC, SA, GRASP.'],
  ];
  items.forEach((it, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.55 + col * 4.15, y = 1.8 + row * 2.55;
    s.addShape('roundRect', { x, y, w: 3.95, h: 2.3, rectRadius: 0.08, fill: { color: C.lgrey }, line: { type: 'none' } });
    s.addText(it[0], { x: x + 0.2, y: y + 0.2, w: 0.9, h: 0.65, fontSize: 26, bold: true, color: '8FA9DE' });
    s.addText(it[1], { x: x + 1.0, y: y + 0.2, w: 2.7, h: 0.45, fontSize: 15, bold: true, color: C.navy });
    s.addText(it[2], { x: x + 0.2, y: y + 0.75, w: 3.55, h: 1.45, fontSize: 13, color: C.black, lineSpacing: 16 });
  });
  footer(s);
}

// ============================================================ 11. PRIMARY RESULT
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Primary result', C.teal);
  slideTitle(s, 'ABC beats Greedy on 16 independent WS instances');

  img(s, 'figures/fig8_forest_WS16.png', 6.8, 4.6, 0.55, 1.85);

  const stats = [
    ['Comparison', 'Mean diff', 'Wins', 'p'],
    ['ABC − Greedy (primary)', '+4.57', '14/16', '3.1e-4'],
    ['SA − Greedy', '+5.99', '15/16', '1.5e-4'],
    ['ABC − SA', '−1.41', '6/16', '0.093'],
    ['ABC − CELF', '+2.69', '16/16', '3.1e-5'],
  ];
  s.addTable(stats.map((r, i) => i === 0 ? r.map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy } } })) : r), {
    x: 7.55, y: 1.85, w: 5.2, h: 2.4,
    fontFace: 'Calibri', fontSize: 13,
    colW: [2.8, 1.3, 1.0, 1.1],
    color: C.black,
    valign: 'middle',
    border: { pt: 0.75, color: 'DFE3EE' },
    fill: { color: C.lgrey },
  });
  s.addText('mean ABC − Greedy 95% CI: [+2.46, +6.91]; p = 3.1×10⁻⁴ passes Bonferroni.', {
    x: 7.55, y: 4.35, w: 5.2, h: 0.6, fontSize: 13, color: C.grey, lineSpacing: 15,
  });
  noteBox(s, 'SA ≥ ABC is stated plainly, not buried. ABC\'s contribution is budget efficiency and a ≥CELF guarantee, not superiority over every set-level searcher.', 7.55, 5.05, 5.2, 1.25, { fill: 'E6F5F1' });
  footer(s);
}

// ============================================================ 12. FAMILIES + REGIME
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Other families & regime');
  slideTitle(s, 'Where the advantage lives');

  s.addText('Family means on WS', { x: 0.55, y: 1.7, w: 6.0, h: 0.35, fontSize: 15, bold: true, color: C.navy });
  s.addChart('bar', [{
    name: 'held-out spread',
    labels: ['Greedy', 'BFS', 'Dense', 'ABC', 'SA'],
    values: [20.0, 13.6, 12.9, 24.6, 26.0],
  }], {
    x: 0.55, y: 2.05, w: 6.0, h: 2.6,
    chartColors: [C.red, C.grey, C.grey, C.teal, C.blue], showValue: true, dataLabelFontSize: 10,
  });

  s.addText('ABC − Greedy gain vs (k − k*)', { x: 7.05, y: 1.7, w: 5.8, h: 0.35, fontSize: 15, bold: true, color: C.navy });
  s.addChart('bar', [{
    name: 'mean abs. gain',
    labels: ['−10', '−5', '0 (k*)', '+5', '+10', '+15'],
    values: [8.42, 2.84, 5.53, 2.16, 4.53, 3.13],
  }], {
    x: 7.05, y: 2.05, w: 5.8, h: 2.6,
    chartColors: [C.teal], showValue: true, dataLabelFontSize: 10,
  });

  const chips = [
    ['+64.6%', 'mean relative gain below k*'],
    ['+23.0%', 'at and above k*'],
    ['≈ 0', 'on BA: Greedy already cascades at k = 5'],
  ];
  chips.forEach((c, i) => {
    const x = 0.55 + i * 4.2;
    s.addShape('roundRect', { x, y: 4.95, w: 3.95, h: 1.4, rectRadius: 0.08, fill: { color: i === 2 ? 'FBEAE8' : C.lgrey }, line: { type: 'none' } });
    s.addText(c[0], { x: x + 0.2, y: 5.1, w: 3.55, h: 0.65, fontSize: 28, bold: true, color: i === 2 ? C.red : C.teal });
    s.addText(c[1], { x: x + 0.2, y: 5.75, w: 3.55, h: 0.55, fontSize: 13, color: C.black, lineSpacing: 14 });
  });
  footer(s);
}

// ============================================================ 13. BUDGET & COST
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Budget & cost');
  slideTitle(s, 'ABC reaches its ceiling at 1× budget; SA needs 4×');

  s.addChart('line', [
    { name: 'ABC ws1000', labels: ['0.5×', '1×', '4×', '8×'], values: [36.13, 37.45, 37.45, 37.45] },
    { name: 'SA ws1000', labels: ['0.5×', '1×', '4×', '8×'], values: [35.17, 30.66, 37.59, 35.59] },
    { name: 'Greedy ws1000', labels: ['0.5×', '1×', '4×', '8×'], values: [27.49, 27.49, 27.49, 27.49] },
    { name: 'ABC/SA ba500', labels: ['0.5×', '1×', '4×', '8×'], values: [93.30, 93.30, 93.95, 93.95] },
  ], {
    x: 0.55, y: 1.85, w: 8.0, h: 4.5,
    chartColors: [C.teal, C.blue, C.red, C.grey],
    lineDataSymbol: 'circle',
    lineDataSymbolSize: 7,
  });

  s.addText('Wall-clock per config', { x: 8.75, y: 1.85, w: 4.0, h: 0.35, fontSize: 15, bold: true, color: C.navy });
  s.addChart('bar', [{
    name: 'seconds',
    labels: ['CELF', 'CELF+LS', 'ABC', 'Greedy', 'GRASP', 'SA'],
    values: [0.18, 0.74, 4.5, 5.96, 6.04, 10.71],
  }], {
    x: 8.75, y: 2.2, w: 4.0, h: 4.15,
    chartColors: ['999999'],
    barDir: 'bar',
    showValue: true,
    dataLabelFontSize: 10,
    dataLabelPosition: 'outEnd',
  });
  footer(s);
}

// ============================================================ 14. DELTA SWEEP CONFOUND
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'The δ-sweep — and its confound');
  slideTitle(s, 'Advantage vs threshold dispersion: real, then a caution');

  s.addChart('line', [
    { name: 'ABC vs Greedy (ws1000)', labels: ['0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5'], values: [75.0, 27.3, 34.8, 24.4, 53.2, -1.2, -3.4, -0.1] },
  ], {
    x: 0.55, y: 1.85, w: 6.0, h: 4.0,
    chartColors: [C.teal], lineDataSymbol: 'circle',
  });
  s.addText('ws1000: large for δ ≤ 0.2, gone beyond', { x: 0.55, y: 5.95, w: 6.0, h: 0.35, fontSize: 13, color: C.grey, align: 'center' });

  s.addChart('line', [
    { name: 'vs CELF', labels: ['0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5'], values: [54.4, 41.1, 38.7, 39.8, 10.7, 2.8, 0.6, 0.4] },
    { name: 'vs Greedy', labels: ['0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5'], values: [2.9, 0.6, -1.7, -0.5, -1.9, -0.9, -0.5, 0.4] },
  ], {
    x: 6.85, y: 1.85, w: 6.0, h: 4.0,
    chartColors: [C.orange, C.teal], lineDataSymbol: 'circle',
  });
  s.addText('ba500: only vs CELF tracks δ; vs Greedy is a wash', { x: 6.85, y: 5.95, w: 6.0, h: 0.35, fontSize: 13, color: C.grey, align: 'center' });

  noteBox(s, 'Confound, controlled: partial correlation removing δ kills every association (p ≥ 0.31). The structural claims rest on the proof, scans and CELF audit — not on this sweep.', 0.55, 6.4, 12.2, 0.55, { fill: 'FDF3E7', line: C.orange, color: '6B4A16' });
  footer(s);
}

// ============================================================ 15. SIGNED EXTENSION
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Signed graphs');
  slideTitle(s, 'A real extension — and a control for the plateau');

  s.addShape('roundRect', { x: 0.55, y: 1.85, w: 5.8, h: 2.0, rectRadius: 0.08, fill: { color: C.navy }, line: { type: 'none' } });
  s.addText('Non-monotonicity witness', { x: 0.85, y: 2.0, w: 5.2, h: 0.35, fontSize: 14, bold: true, color: C.teal });
  s.addText('σ({1,2,3,4}) = 6  >  σ({1,2,3,4,5}) = 5', { x: 0.85, y: 2.45, w: 5.2, h: 0.55, fontSize: 20, bold: true, color: C.white });
  s.addText('Seeding an enemy leaf blocks the hub: a larger seed set can have smaller influence.', { x: 0.85, y: 3.05, w: 5.2, h: 0.65, fontSize: 13, color: C.ice, lineSpacing: 15 });

  const kpis = [['17.6', 'ABC mean'], ['17.7', 'SA mean'], ['15.4', 'Greedy'], ['14.6', 'CELF']];
  kpis.forEach((k, i) => {
    const x = 0.55 + i * 1.55;
    const col = i < 2 ? C.teal : (i === 2 ? C.navy : C.red);
    s.addShape('roundRect', { x, y: 4.05, w: 1.4, h: 1.3, rectRadius: 0.08, fill: { color: C.lgrey }, line: { type: 'none' } });
    s.addText(k[0], { x, y: 4.15, w: 1.4, h: 0.75, align: 'center', fontSize: 24, bold: true, color: col });
    s.addText(k[1], { x, y: 4.85, w: 1.4, h: 0.4, align: 'center', fontSize: 11, color: C.grey });
  });

  s.addText('Sign-ablation control', { x: 6.7, y: 1.85, w: 6.0, h: 0.35, fontSize: 15, bold: true, color: C.navy });
  s.addChart('bar', [
    { name: 'Greedy', labels: ['sbm120 signed', 'sbm120 twin', 'ws300 signed', 'ws300 twin'], values: [19.9, 19.8, 21.0, 36.8] },
    { name: 'ABC', labels: ['sbm120 signed', 'sbm120 twin', 'ws300 signed', 'ws300 twin'], values: [38.9, 55.6, 26.9, 40.7] },
  ], {
    x: 6.7, y: 2.25, w: 6.0, h: 3.0,
    chartColors: [C.red, C.teal], showValue: true, dataLabelFontSize: 9, barGrouping: 'clustered',
  });
  s.addText('The ABC-vs-Greedy advantage persists — and grows — when signs are removed. The plateau effect, not the signs, drives the gain.', {
    x: 6.7, y: 5.4, w: 6.0, h: 0.7, fontSize: 13, color: C.black, lineSpacing: 15,
  });
  footer(s);
}

// ============================================================ 16. IMPLEMENTATION STORY
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Implementation story');
  slideTitle(s, 'From the first implementation to the final submission');

  const cols = [
    ['Original implementation', 'Standard Greedy + CELF pipeline, Linear Threshold assumptions, graph as (n,k) row, budgets not audited, no held-out scoring.', C.lgrey],
    ['Shortcomings found', 'CELF lazy-bound failed 50% of re-checks; submodularity false; small-graph wins vanished under equal budgets and fresh thresholds.', 'FBEAE8'],
    ['Fixes / optimisations', 'Equal-budget protocol, held-out scoring, graph = statistical unit. Canonical ABC → gbest-guided, tournament, directed scouts, dedupe, Φ tie-break, CELF-seeded population. Refinement removed; ABC-CELF kept only as ablation.', 'E6F5F1'],
    ['Where we failed', 'Φ as global fitness hurt; Greedy+Φ trailed Greedy; Metropolis variant stayed below SA; δ-sweep correlation confounded; "signed gives ABC a distinct advantage" rejected by sign-ablation control.', 'FDF3E7'],
    ['Final submission', 'Monotone proof + non-submodular witness + CELF audit + LOO-selected ABC + honest equal-budget evaluation (WS/BA/SBM) + signed extension. ABC is a budget-efficient, guarantee-carrying set-level searcher — not the best possible one.', C.lgrey],
  ];
  const w = 2.42, gap = 0.12, x0 = 0.55, y0 = 1.75, h = 4.95;
  cols.forEach(([title, body, fill], i) => {
    const x = x0 + i * (w + gap);
    s.addShape('roundRect', { x, y: y0, w, h, rectRadius: 0.07, fill: { color: fill }, line: { type: 'none' } });
    s.addText(title, { x: x + 0.12, y: y0 + 0.15, w: w - 0.24, h: 0.55, fontSize: 13, bold: true, color: C.navy, lineSpacing: 14 });
    s.addText(body, { x: x + 0.12, y: y0 + 0.75, w: w - 0.24, h: h - 0.95, fontSize: 11.5, color: C.black, lineSpacing: 14 });
  });
  footer(s);
}

// ============================================================ 17. CONCLUSIONS
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  sectionTag(s, 'Conclusion', C.teal);
  s.addText('Five claims that survive scrutiny — and three that don\'t', {
    x: 0.7, y: 1.0, w: 11.8, h: 0.9, fontSize: 30, bold: true, color: C.white,
  });

  const supports = [
    '1. The model is monotone (proved) and not submodular (witnessed, scanned, explained via Mossel–Roch).',
    '2. CELF breaks: 50.2% lazy violations; differs from Greedy in 16/17 configs, strictly worse in 13/17.',
    '3. On sparse clustered graphs below the contagion threshold, set-level search beats Greedy: +4.6 held-out on 16 WS; ties on BA; beats CELF everywhere.',
    '4. ABC is a valid set-level searcher — budget-efficient (ceiling at 1× vs SA\'s 4×), ≥CELF guarantee, CV 1.7%.',
    '5. The signed extension adds the model and a non-monotonicity witness — no distinct search advantage.',
  ];
  s.addText('What the evidence supports', { x: 0.7, y: 2.05, w: 11.8, h: 0.4, fontSize: 14, bold: true, color: C.teal });
  let y = 2.55;
  supports.forEach((t) => {
    s.addShape('ellipse', { x: 0.7, y: y + 0.08, w: 0.16, h: 0.16, fill: { color: C.teal }, line: { type: 'none' } });
    s.addText(t, { x: 1.05, y: y - 0.1, w: 11.4, h: 0.6, fontSize: 14, color: 'E6EAF5', lineSpacing: 16 });
    y += 0.62;
  });

  const donts = [
    '✗  “ABC is the best set-level searcher” — SA ≥ ABC on WS and BA.',
    '✗  “The advantage tracks submodularity” — the δ-sweep correlation is confounded.',
    '✗  “Signed graphs need ABC specifically” — sign-ablation control shows it is the plateau effect.',
  ];
  y += 0.1;
  s.addText('What not to claim', { x: 0.7, y, w: 11.8, h: 0.4, fontSize: 14, bold: true, color: C.red2 });
  y += 0.5;
  donts.forEach((t) => {
    s.addText(t, { x: 0.7, y, w: 11.4, h: 0.5, fontSize: 13.5, color: 'F5D5D0', lineSpacing: 15 });
    y += 0.48;
  });
  footer(s, true);
}

// ============================================================ 18. FUTURE SCOPE
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  sectionTag(s, 'Future scope');
  slideTitle(s, 'Limitations and what comes next');

  const cols = [
    ['Current limitations', 'Absolute effects are small (<1% of n among top methods). BA/SBM evidence is exploratory (n=8). Ceiling certification rests on budget-insensitivity; sandwich bound is vacuous at this scale. Candidate pool capped at 200 nodes on large graphs.'],
    ['Next steps', 'Real-network signed tests (Epinions, Slashdot, Bitcoin-Alpha, ego-Facebook, email-Eu-core). Tighter ceiling certification. Budget-aware k selection relative to contagion threshold k*. Extend the audit to other non-submodular diffusion families.'],
    ['Final framing', 'The submission shows that for majority-rule diffusion on sparse, clustered graphs, coalitions carry the influence and a carefully audited set-level searcher is a sound, defensible choice.'],
  ];
  const w = 4.05, gap = 0.2, x0 = 0.55, y0 = 1.75, h = 4.6;
  cols.forEach(([title, body], i) => {
    const x = x0 + i * (w + gap);
    s.addShape('roundRect', { x, y: y0, w, h, rectRadius: 0.08, fill: { color: C.lgrey }, line: { type: 'none' } });
    s.addText(title, { x: x + 0.15, y: y0 + 0.15, w: w - 0.3, h: 0.45, fontSize: 15, bold: true, color: C.navy });
    s.addText(body, { x: x + 0.15, y: y0 + 0.65, w: w - 0.3, h: h - 0.85, fontSize: 13.5, color: C.black, lineSpacing: 17 });
  });
  footer(s);
}

// ============================================================ WRITE OUT
const outPath = path.join(__dirname, '..', 'Majority_Rule_ABC_Editable_Deck.pptx');
pres.writeFile({ fileName: outPath })
  .then(() => {
    const stats = fs.statSync(outPath);
    console.log(`Wrote ${outPath}`);
    console.log(`Slides: 18 | Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
  })
  .catch((err) => { console.error(err); process.exit(1); });
