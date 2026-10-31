import React from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, BarChart, Bar, ReferenceLine, LabelList, Cell,
} from 'recharts'
import { Head, Foot, KPI, Table, Panel, Forest, Legend } from './ui.jsx'
import {
  C, sweepK13, structureTable, exactTable, looTable,
  ws16, famStats, regime, budgetData, wallClock,
  deltaBa, deltaWs, signControl,
} from './data.js'

const grid = '#e4e2da'
const tick = { fontSize: 12.5, fill: '#8a93a3' }

const tooltipStyle = {
  fontSize: 13.5, borderRadius: 8, border: '1px solid #e2e0d8',
  boxShadow: '0 6px 18px rgba(16,20,28,.12)',
}

export const S1Title = () => (
  <div className="slide title-slide">
    <div className="wrap">
      <div className="overline">Influence Maximisation · Technical Presentation</div>
      <h1>
        The majority-rule model is <span className="accent">monotone, not submodular</span> —
        and that is why <span className="accent">set-level search</span> beats greedy
      </h1>
      <div className="lede">
        Machine-checked structure · a literature-driven Artificial Bee Colony optimiser ·
        equal-budget, held-out evaluation with the graph as the statistical unit ·
        a signed-graph extension
      </div>
      <div className="meta">
        <span className="chip on">proved: monotone ✓</span>
        <span className="chip red">disproved: submodular ✗</span>
        <span className="chip">16 WS instances · p = 3.1×10⁻⁴</span>
        <span className="chip grey">R = 1000 held-out scoring</span>
      </div>
    </div>
    <Foot src="results/s10_ws16_stats.json · summary.json" n={16} i={0} />
  </div>
)

export const S2Model = () => (
  <div className="slide">
    <Head
      kicker="1 · The model"
      title="Progressive threshold diffusion — one rule, four variants"
      subtitle="Thresholds are drawn once and frozen; all evaluations share common random numbers (reproducible, noise-free comparisons)."
      tag="Eq. 1"
    />
    <div className="body">
      <div className="split">
        <div className="col gap14 grow">
          <div className="formula">
            <span className="hl">v activates</span> ⟺ |{'{ active neighbours of v }'}|{' '}
            <span className="hl">&gt;</span> θ<sub>v</sub> · deg(v)<br />
            <span style={{ color: '#9fb4c9' }}>σ(S) = E_θ[ |A∞(S, θ)| ]</span>
          </div>
          <div>
            <div className="h3"><span className="num">▶</span>Variants along one δ-dial</div>
            <div className="chips">
              <span className="chip on">dmr · θ = ½ — strict majority</span>
              <span className="chip on">smr · θ ~ U[½−δ, ½+δ] — workhorse δ = 0.15</span>
              <span className="chip">lt · θ ~ U[0,1] — submodular control</span>
              <span className="chip">slt · θ ~ U[0, ½+δ] — sandwich upper bound</span>
            </div>
          </div>
          <div>
            <div className="h3"><span className="num">▶</span>Estimator</div>
            <p className="p">
              Monte-Carlo over <b>R = 150–200</b> frozen threshold vectors (fit),{' '}
              <b>R = 1000</b> fresh draws (held-out score). Cascades batched as sparse
              matrix products — <b>GPU-accelerated</b> (RTX 3060, automatic CPU fallback).
            </p>
          </div>
          <div className="note">
            <b>Signed extension (§10):</b> edges carry sign + weight; v activates iff{' '}
            (positive active weight − negative active weight) &gt; θ<sub>v</sub>·deg(v).
            All-positive ⟹ exactly this model. Negative edges break{' '}
            <b>monotonicity</b> — verified witnesses ahead.
          </div>
        </div>
        <div className="col gap10" style={{ width: 470 }}>
          <Panel title="Thresholds in action — the strict majority (dmr)">
            <div className="small" style={{ marginBottom: 8 }}>
              θ = ½ everywhere. A node of degree d needs <b>⌊d/2⌋ + 1</b> active neighbours.
            </div>
            <div className="chips" style={{ marginBottom: 10 }}>
              <span className="chip">deg 1 → needs 1</span>
              <span className="chip">deg 3 → needs 2</span>
              <span className="chip">deg 5 → needs 3</span>
              <span className="chip on">coalition, not individuals</span>
            </div>
            <div className="p">
              The last neighbour before the threshold is worth <b>everything</b>; the ones
              before it are worth <b>nothing</b>. This is increasing returns in disguise —
              the whole story of this deck.
            </div>
          </Panel>
          <Panel title="Why common random numbers matter">
            <p className="p">
              σ is submodular only <b>in expectation</b>. Reusing the same frozen draws
              across candidate sets removes MC noise from every marginal-gain comparison
              — and makes the entire study exactly reproducible.
            </p>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="majority_rule_model.py" n={16} i={1} />
  </div>
)

export const S3Structure = () => (
  <div className="slide">
    <Head
      kicker="2 · Structure"
      title="Monotone — proven. Submodular — false, with a two-line witness"
      subtitle="Mossel–Roch: submodular local functions suffice for submodularity; a majority CDF is convex below ½, so the condition fails — and the scans confirm actual violations."
      tag="§2.1–2.4"
    />
    <div className="body">
      <div className="split">
        <div className="col gap12 grow">
          <Panel title="Monotonicity — lock-step induction (5,800 pairs, 0 violations)">
            <div className="formula" style={{ fontSize: 16.5 }}>
              S ⊆ T ⟹ A<sub>t</sub>(S) ⊆ A<sub>t</sub>(T) ∀t ⟹ σ(S) ≤ σ(T){' '}
              <span className="hl">∎</span>
            </div>
            <div className="small" style={{ marginTop: 10 }}>
              If v fires under S, its active-neighbour count under T is only larger —
              pointwise in θ, then take expectations.
            </div>
          </Panel>
          <Panel title="Submodularity — the minimal counterexample (K₁,₃, dmr)">
            <Table
              cols={['coalition', 'gain of adding y', 'verdict']}
              rows={[
                { 'coalition': 'S = ∅', 'gain of adding y': 'σ({y}) − σ(∅) = 1.0', 'verdict': { v: 'y alone flips nothing', cls: '' } },
                { 'coalition': 'T = {x}', 'gain of adding y': 'σ({x,y}) − σ({x}) = 3.0', 'verdict': { v: 'x,y flip w; w flips z', cls: 'pos' } },
              ]}
            />
            <div className="small" style={{ marginTop: 8 }}>
              Bigger set, <b>bigger</b> marginal gain — increasing returns.{' '}
              <span className="cross">violation 3.0 − 1.0 = +2.0</span>
            </div>
          </Panel>
        </div>
        <div className="col grow">
          <Panel title="Does softening the threshold rescue submodularity? (R = 20k)">
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={sweepK13} margin={{ top: 8, right: 14, left: -18, bottom: 2 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="d" tick={tick} label={{ value: 'threshold dispersion δ', position: 'insideBottom', offset: -2, fontSize: 12.5, fill: '#8a93a3' }} />
                <YAxis tick={tick} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}`, '']} />
                <ReferenceLine y={0} stroke="#9aa3b3" />
                <Line type="monotone" dataKey="k13" name="K₁,₃ (T = 1 leaf)" stroke={C.red} strokeWidth={2.4} dot={{ r: 3.5 }} connectNulls />
                <Line type="monotone" dataKey="k19" name="K₁,₉ (T = 4 leaves)" stroke={C.blue} strokeWidth={2.4} dot={{ r: 3.5 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <div className="small" style={{ marginTop: 6 }}>
              <b>Violation magnitude (gain_T − gain_S).</b> K₁,₃ clear for δ ≲ 0.3;
              larger stars keep it visible to δ = 0.40. At δ = 0.5 (LT) it vanishes —
              the submodular control.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="theory_checks.py · s1_structure.json" n={16} i={2} />
  </div>
)

export const S4Scan = () => (
  <div className="slide">
    <Head
      kicker="3 · Exhaustive scan"
      title="Every majority variant violates submodularity; the LT control is clean"
      subtitle="All (S ⊆ T, v ∉ T) triples with |T| ≤ 3, noise tolerance calibrated from σ’s own sampling error."
      tag="Table 1"
    />
    <div className="body">
      <div className="col grow gap10">
        <Table
          cols={['graph', 'variant', 'monotone', 'raw viol. rate', 'non-deg. viol. rate', 'γ_min']}
          rows={structureTable.map((r) => ({
            graph: r.g,
            variant: r.v,
            monotone: { v: '✓', cls: 'ok' },
            'raw viol. rate': r.ctl ? { v: r.viol, cls: 'ok' } : r.viol,
            'non-deg. viol. rate': r.ctl ? { v: r.nd, cls: 'ok' } : r.nd,
            'γ_min': r.gam.toFixed(3),
          }))}
          rowClass={(r) => (r.variant === 'LT (0.50)' ? 'hl' : '')}
          caption="Monotonicity verified on 600/600/250 sampled pairs per configuration (5,800 total): zero violations."
        />
        <div className="row">
          <KPI tone="teal" v="0.29 – 0.85" l="γ_min across majority settings — the honest (optimistic) replacement for the (1−1/e) guarantee" />
          <KPI tone="blue" v="0.96 – 0.99" l="γ_min for the LT control — submodular, as theory demands" />
          <KPI tone="red" v="0.0%" l="violation rate for LT — the calibration check" />
        </div>
      </div>
      <div className="col gap12" style={{ width: 380 }}>
        <Panel title="Why: the local activation function">
          <div className="p" style={{ fontSize: 15 }}>
            f<sub>v</sub>(a) = Pr[θ &lt; a/deg] = <b>F_θ(a/deg)</b>. Submodular local functions
            ⟺ <b>concave CDF</b>. Majority CDFs are S-curves — convex on the lower stretch:
            the first ⌊d/2⌋ neighbours are worthless, the next one is worth everything.
          </div>
          <div className="chips" style={{ marginTop: 10 }}>
            <span className="chip on">LT: F(x) = x — concave ✓</span>
            <span className="chip red">majority: S-curve — convex ✗</span>
          </div>
        </Panel>
        <Panel title="Corollary — the classical guarantee is void">
          <div className="p" style={{ fontSize: 15 }}>
            Greedy’s (1−1/e) bound requires submodularity. <b>Hardness:</b> Lu et al. (JOCO
            2012) rule out n<sup>1−ε</sup>-approximation for <i>general</i> deterministic
            thresholds; our uniform-½ rule is a special case, so hardness does not
            transfer automatically — quality is measured empirically instead.
          </div>
        </Panel>
      </div>
    </div>
    <Foot src="results/s1_structure.csv" n={16} i={3} />
  </div>
)

export const S5Celf = () => (
  <div className="slide">
    <Head
      kicker="4 · Why greedy-style methods fail"
      title="CELF’s lazy bound is wrong half the time — measurably"
      subtitle="CELF prunes with stale gains as upper bounds. Under a majority rule gains increase — the bound is invalid, and the audit counts every break."
      tag="§3 · audit"
    />
    <div className="body">
      <div className="split">
        <div className="col gap12">
          <div className="kpis">
            <KPI tone="red" v="50.2%" l="lazy-bound violations over all CELF runs — 418 of 832 re-checks" />
            <KPI tone="red" v="16 / 17" l="configs where CELF differs from exact Greedy" />
            <KPI tone="red" v="13 / 17" l="configs where CELF is strictly worse (held-out)" />
          </div>
          <Panel title="The collapse on hub graphs (ba500, k = 30, held-out)">
            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={[
                { a: 'CELF', s: 131.9, fill: C.red },
                { a: 'Degree', s: 473.7, fill: C.grey },
                { a: 'Greedy', s: 491.5, fill: C.orange },
                { a: 'ABC', s: 493.8, fill: C.teal },
              ]} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="a" tick={tick} />
                <YAxis tick={tick} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="s" name="held-out spread" radius={[5, 5, 0, 0]}>
                  <Cell fill={C.red} />
                  <Cell fill={C.grey} />
                  <Cell fill={C.orange} />
                  <Cell fill={C.teal} />
                  <LabelList dataKey="s" position="top" style={{ fontSize: 12.5, fontWeight: 700, fill: '#182130' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="small">
              CELF loses to a <b>trivial degree ranking</b> — a +273% gap that is CELF’s
              collapse, not ABC’s miracle. All relative gains against CELF should be read
              this way.
            </div>
          </Panel>
        </div>
        <div className="col gap12">
          <Panel title="Mechanism">
            <div className="p" style={{ fontSize: 15.5 }}>
              Greedy (and CELF) pick the node with the largest <b>singleton</b> marginal
              gain. On a majority plateau those gains are ≈ 0 for everyone — greedy is
              choosing among ties, and CELF additionally <b>prunes the wrong way</b>:
              its stale gains are not upper bounds anymore.
            </div>
            <div className="chips" style={{ marginTop: 12 }}>
              <span className="chip on">influence lives in coalitions</span>
              <span className="chip red">marginal search is blind to them</span>
            </div>
          </Panel>
          <Panel title="What survives of the theory">
            <div className="p" style={{ fontSize: 15.5 }}>
              <b>Sandwich bound</b> (Lu–Chen–Lakshmanan style): the concave envelope of the
              majority CDF is F̄(x) = min(x/(½+δ), 1) — an LT-type model that dominates σ
              pathwise and <b>is</b> submodular, so CELF is valid on it. Measured α = σ̄/σ ≈ 13
              ⟹ the guarantee is <b>vacuous at small δ</b>. Kept as theory, not as a seeder.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/val_celf_audit.csv · s3_main.csv" n={16} i={4} />
  </div>
)

export const S6Abc = () => (
  <div className="slide">
    <Head
      kicker="5 · The optimiser"
      title="ABC, upgraded along the literature — and pruned by a leave-one-out ablation"
      subtitle="Canonical ABC’s documented weaknesses (W1–W9) each get one targeted operator; 26 tuning configs × 20 seeds on 13 graphs disjoint from the test graphs decide what stays."
      tag="§4 · Table 2"
    />
    <div className="body">
      <div className="split">
        <div className="col grow gap10">
          <Table
            cols={['weakness of canonical ABC', 'literature', 'this work']}
            rows={[
              { 'weakness of canonical ABC': 'weak exploitation', literature: 'gbest-guided ABC (Zhu & Kwong ’10)', 'this work': { v: 'gbest partner, p = 0.6 — the only operator that matters', cls: 'pos' } },
              { 'weakness of canonical ABC': 'roulette premature convergence', literature: 'tournament selection', 'this work': 'tournament, size 4' },
              { 'weakness of canonical ABC': 'blind random scouts', literature: 'directed scouts (Kiran & Fındık ’15)', 'this work': 'rebuild near the incumbent' },
              { 'weakness of canonical ABC': 'fixed perturbation', literature: 'modification rate (Akay & Karaboga ’12)', 'this work': 'n_mod swaps + mutation' },
              { 'weakness of canonical ABC': 'duplicate food sources', literature: '—', 'this work': 'dedupe' },
              { 'weakness of canonical ABC': 'rigid moves on plateaus (ABC+VNS critique)', literature: '—', 'this work': 'Φ tie-break + accept-equal + random moves' },
              { 'weakness of canonical ABC': 'refinement blows the budget', literature: '—', 'this work': { v: 'removed — it hurt (LOO)', cls: 'neg' } },
              { 'weakness of canonical ABC': '1-opt can’t repair coalitions', literature: '—', 'this work': '2-swap + cascade-aware candidates (in CELF+LS)' },
            ]}
          />
          <div className="warn">
            <b>Negative result, stated once:</b> Φ(S) = Σ min(1, net/need) as a global
            fitness term hurt (ba500 51.9 vs 57.5 held-out); as a tie-break it is
            neutral; Greedy+Φ trails Greedy on WS. A Metropolis variant helps on WS
            (+2.5) but stays below SA — flag kept, not default.
          </div>
        </div>
        <div className="col gap12" style={{ width: 420 }}>
          <Panel title="Leave-one-out ablation (Δ held-out vs full config)">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart layout="vertical" data={looTable.map((r) => ({ ...r, d: typeof r.d === 'number' ? r.d : 0, sig: r.sig }))} margin={{ top: 4, right: 18, left: 4, bottom: 0 }}>
                <CartesianGrid stroke={grid} horizontal={false} />
                <XAxis type="number" tick={tick} domain={[-0.16, 0.14]} />
                <YAxis type="category" dataKey="op" width={150} tick={{ fontSize: 12, fill: '#4a5468' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <ReferenceLine x={0} stroke="#9aa3b3" />
                <Bar dataKey="d" name="Δ mean held-out" radius={[0, 5, 5, 0]}>
                  {looTable.map((r) => (
                    <Cell key={r.op} fill={r.sig ? (r.d < 0 ? C.teal : C.red) : '#cfd4dc'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="small">
              <b>−gbest</b>: −0.13, p = 6×10⁻⁴ (keep). <b>−reserve</b>: +0.12, p = 0.012 (drop).
              Everything else: &lt;1% and non-significant — honest minimal configuration.
            </div>
          </Panel>
          <div className="note">
            <b>Final method = ABC (proposed):</b> CELF-seeded, gbest-guided population
            search, <b>no refinement phase</b>. Elitism ⟹ never worse than CELF. The
            “ABC-CELF hybrid” is kept only as an ablation.
          </div>
        </div>
      </div>
    </div>
    <Foot src="results/val_loo.csv" n={16} i={5} />
  </div>
)

export const S7Protocol = () => (
  <div className="slide">
    <Head
      kicker="6 · Evaluation protocol"
      title="Equal budgets, held-out scoring, and the graph as the statistical unit"
      subtitle="Every budgeted solver gets Greedy’s exact cost (k × |pool|). Final seed sets are re-scored on 1,000 fresh threshold draws."
      tag="v3 methodology"
    />
    <div className="body">
      <div className="split-3">
        <Panel title="1 · Equal budget">
          <div className="p" style={{ fontSize: 15.5 }}>
            Shared budget = <b>k × |pool|</b> fresh cascade evaluations — Greedy’s natural
            cost. CELF runs as the cheap baseline. Audit: <b>0 violations</b> across 532
            rows of the main study. No more “ABC got more evaluations” confound.
          </div>
        </Panel>
        <Panel title="2 · Held-out scoring">
          <div className="p" style={{ fontSize: 15.5 }}>
            Fit on R = 150 frozen draws; score every final seed set on{' '}
            <b>independent draws, R = 1000</b>. In-sample minus held-out gap for ABC:
            +0.13 (worst +4.4) — conclusions are not overfitting artefacts. The OPT set
            itself is re-scored, fixing best-of-many bias.
          </div>
        </Panel>
        <Panel title="3 · Graph = unit">
          <div className="p" style={{ fontSize: 15.5 }}>
            Spreads are averaged over k <b>within</b> a graph, then compared across{' '}
            <b>16–8 independent instances per family</b> (varying generator seeds and
            degree/attachment/density). Wilcoxon + cluster-bootstrap CIs. Correlated
            k-configs cannot masquerade as independent evidence.
          </div>
        </Panel>
      </div>
      <div className="row" style={{ marginTop: 18 }}>
        <div className="col grow">
          <Panel title="4 · Pre-specified primary + multiplicity" style={{ height: '100%' }}>
            <div className="p" style={{ fontSize: 15.5 }}>
              Exactly one comparison is declared primary:{' '}
              <b>ABC vs Greedy on WS (n = 16 graphs)</b>. Nine family tests ⟹ Bonferroni
              threshold <b>0.0056</b>. Everything else is exploratory: effect sizes with
              CIs, not p-values. Forest plots show every per-graph difference.
            </div>
          </Panel>
        </div>
        <div className="col grow">
          <Panel title="5 · Baselines, so “set-level search” is a real claim" style={{ height: '100%' }}>
            <div className="chips">
              <span className="chip">zero-MC: Degree · PageRank · k-Core · BFSBall · DenseCluster</span>
              <span className="chip">marginal: Greedy · Greedy+Φ · CELF · CELF+LS · Sandwich-CELF</span>
              <span className="chip on">set-level: ABC · SA · GRASP</span>
            </div>
            <div className="small" style={{ marginTop: 10 }}>
              SA uses the <b>same swap neighbourhood</b> as ABC (1-swap, geometric cooling,
              reheat) — the comparison isolates the search strategy, not the representation.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="run_experiments.py (v3) · summary.json" n={16} i={6} />
  </div>
)

export const S8Primary = () => (
  <div className="slide">
    <Head
      kicker="7 · Primary result"
      title="On 16 independent Watts–Strogatz instances, ABC beats Greedy by +4.6 held-out"
      subtitle="Graph-averaged over k ∈ {5, 10, 20}; paired across graphs; cluster-bootstrap 95% CIs. The primary comparison passes Bonferroni."
      tag="p = 3.1×10⁻⁴ ✓"
    />
    <div className="body">
      <div className="split">
        <div className="col gap12 grow">
          <Panel title="Per-graph paired differences (held-out spread, k-averaged)">
            <Forest
              rows={ws16.forest}
              xlab="Δ held-out spread  (teal = ABC − Greedy · blue = ABC − SA)"
              width={470}
              ci={ws16.primary.ci}
              min={-8}
              max={15}
            />
            <div className="small" style={{ marginTop: 4 }}>
              Teal markers right of zero: <b>14 of 16</b> graphs favour ABC over Greedy.
              Blue markers: SA − ABC positive on <b>10 of 16</b> — see the caveat panel.
            </div>
          </Panel>
        </div>
        <div className="col gap12" style={{ width: 470 }}>
          <div className="kpis">
            <KPI tone="teal" v="+4.57" l="mean ABC − Greedy · CI [+2.46, +6.91] · 14/16 wins · p = 3.1×10⁻⁴" />
            <KPI tone="blue" v="+5.99" l="mean SA − Greedy · CI [+4.01, +7.85] · 15/16 · p = 1.5×10⁻⁴" />
          </div>
          <Table
            cols={['comparison', 'mean diff', 'CI (bootstrap)', 'wins', 'p']}
            rows={[
              { 'comparison': { v: 'ABC − Greedy (primary)', cls: 'pos' }, 'mean diff': { v: '+4.57', cls: 'pos' }, 'CI (bootstrap)': '[+2.46, +6.91]', wins: '14/16', p: { v: '3.1e-4', cls: 'pos' } },
              { 'comparison': 'SA − Greedy', 'mean diff': '+5.99', 'CI (bootstrap)': '[+4.01, +7.85]', wins: '15/16', p: '1.5e-4' },
              { 'comparison': { v: 'ABC − SA', cls: 'neg' }, 'mean diff': { v: '−1.41', cls: 'neg' }, 'CI (bootstrap)': '[−2.84, −0.12]', wins: '6/16', p: '0.093' },
              { 'comparison': 'ABC − CELF', 'mean diff': '+2.69', 'CI (bootstrap)': '[+1.83, +3.89]', wins: '16/16', p: '3.1e-5' },
            ]}
            caption="SA is at least as good as ABC — stated plainly, not buried."
          />
          <Panel title="Draws-based cascade metric — fraction of held-out draws with spread ≥ k + 10">
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={Object.entries(ws16.cascade).map(([k, v]) => ({ a: k, s: +(v * 100).toFixed(1) }))} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="a" tick={tick} />
                <YAxis tick={tick} unit="%" />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, 'draws ≥ k+10']} />
                <Bar dataKey="s" radius={[5, 5, 0, 0]}>
                  <Cell fill={C.red} />
                  <Cell fill={C.grey} />
                  <Cell fill={C.orange} />
                  <Cell fill={C.teal} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="small">
              Greedy triggers a full cascade in only <b>33%</b> of threshold draws; ABC in 54%.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/s10_ws16_stats.json · fig8_forest_WS16.png" n={16} i={7} />
  </div>
)

export const S9Families = () => (
  <div className="slide">
    <Head
      kicker="8 · The other families"
      title="BA is a wash; SBM is suggestive; CELF loses everywhere"
      subtitle="8 fresh instances per family, same protocol. All comparisons exploratory (n = 8) — read CIs, not p-values."
      tag="§8.2–8.3"
    />
    <div className="body">
      <div className="split">
        <div className="col gap12 grow">
          <Panel title="Barabási–Albert × 8 — the wash">
            <div className="kpis" style={{ marginBottom: 10 }}>
              <KPI v="+0.26" l="ABC − Greedy · CI [−0.42, +1.06] · 5/8 · p = 0.64 — a tie" />
              <KPI tone="blue" v="−0.49" l="ABC − SA · CI [−0.79, −0.18] · 2/8 · p = 0.039 — SA favoured" />
              <KPI tone="teal" v="+12.4" l="ABC − CELF · 8/8 · p = 0.0078" />
            </div>
            <div className="small">
              Greedy already cascades at k = 5 on hub graphs — no plateau window, no
              advantage. Family means: SA 40.6 · ABC 40.1 · Greedy 39.8 · CELF 27.7 ·
              BFSBall 22.9. <b>Constructive baselines are nowhere near</b>: search, not
              locality, buys the cascade.
            </div>
          </Panel>
          <Panel title="Stochastic block model × 8 — suggestive">
            <div className="kpis">
              <KPI tone="teal" v="+0.58" l="ABC − Greedy · CI [+0.07, +1.14] · 6/8 · p = 0.078 — suggestive (CI excludes 0)" />
              <KPI v="−0.05" l="ABC − SA · tie (p = 0.84)" />
              <KPI tone="teal" v="+1.57" l="ABC − CELF · 8/8 · p = 0.0078" />
            </div>
            <div className="small" style={{ marginTop: 8 }}>
              Means: SA 13.7 · ABC 13.6 · Greedy 13.1 · CELF 12.1. The community structure
              gives a mild version of the WS plateau effect.
            </div>
          </Panel>
        </div>
        <div className="col gap12" style={{ width: 470 }}>
          <Panel title="Locality is not the story">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { a: 'Greedy', s: 20.0, fill: C.red },
                { a: 'BFS ball', s: 13.6, fill: C.grey },
                { a: 'Dense cluster', s: 12.9, fill: C.grey },
                { a: 'ABC', s: 24.6, fill: C.teal },
                { a: 'SA', s: 26.0, fill: C.blue },
              ]} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="a" tick={tick} />
                <YAxis tick={tick} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="s" name="held-out spread (WS family mean)" radius={[5, 5, 0, 0]}>
                  <Cell fill={C.red} />
                  <Cell fill={C.grey} />
                  <Cell fill={C.grey} />
                  <Cell fill={C.teal} />
                  <Cell fill={C.blue} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="small" style={{ marginTop: 6 }}>
              Zero-simulation constructive seeders (target-set-selection style) trail SA by
              +13.1 on WS (8/8, p = 0.0078). Connected coalitions are <b>necessary</b>,
              evaluations spent on search are what makes them <b>winning</b>.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/s10_multi_stats.json" n={16} i={8} />
  </div>
)

export const S10Regime = () => (
  <div className="slide">
    <Head
      kicker="9 · When the advantage appears"
      title="The gain lives below Greedy’s cascade threshold k*"
      subtitle="k* = smallest k at which Greedy’s set first triggers any cascade. Absolute gains, so the plateau baseline cannot inflate them."
      tag="§8.5"
    />
    <div className="body">
      <div className="split">
        <div className="col grow">
          <Panel title="Absolute ABC − Greedy gain vs (k − k*), WS family">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={regime} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="dk" tick={tick} label={{ value: 'k − k*', position: 'insideBottom', offset: -2, fontSize: 12.5, fill: '#8a93a3' }} />
                <YAxis tick={tick} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`+${v} nodes`, 'mean abs. gain']} />
                <ReferenceLine y={0} stroke="#9aa3b3" />
                <Bar dataKey="gain" radius={[5, 5, 0, 0]} fill={C.teal}>
                  <LabelList dataKey="gain" position="top" formatter={(v) => `+${v}`} style={{ fontSize: 12.5, fontWeight: 700, fill: '#0e7a65' }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="kpis" style={{ marginTop: 8 }}>
              <KPI tone="teal" v="+64.6%" l="mean relative gain below k*" />
              <KPI v="+23.0%" l="at and above k*" />
              <KPI tone="red" v="≈ 0" l="on BA — Greedy cascades at k = 5 already" />
            </div>
          </Panel>
        </div>
        <div className="col gap12" style={{ width: 470 }}>
          <Panel title="The honest reading">
            <div className="p" style={{ fontSize: 15.5 }}>
              Below k*, Greedy is stuck near k itself, so any method that finds a cascade
              wins big — this is a <b>description of Greedy’s blindness</b>, not an
              independent mechanism. We tried to define k* independently by percolation
              (random k-sets): <b>no random k-set ever cascades for k ≤ 20</b>. That is
              itself the phenomenon: majority cascades require <i>chosen</i> coalitions.
            </div>
          </Panel>
          <Panel title="Why this matters for practice">
            <div className="p" style={{ fontSize: 15.5 }}>
              If the budget allows k beyond the contagion threshold, plain Greedy suffices.
              The metaheuristic earns its cost exactly in the <b>sub-critical window</b> —
              which is also where real campaigns (small budgets) operate.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/s10_regime.csv · s10_percolation.csv" n={16} i={9} />
  </div>
)

export const S11Budget = () => (
  <div className="slide">
    <Head
      kicker="10 · Budget & cost"
      title="ABC reaches its ceiling at 1× budget; SA needs 4× to catch up"
      subtitle="ws1000, k = 15, held-out. Greedy’s cost is budget-fixed (its flat line). ABC spends its full budget — the plateau is real, not early stopping."
      tag="§8.6"
    />
    <div className="body">
      <div className="split">
        <div className="col grow">
          <Panel title="Held-out spread vs evaluation budget (× k·|pool|)">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={budgetData} margin={{ top: 8, right: 14, left: -14, bottom: 2 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="mult" tick={tick} label={{ value: 'budget multiplier', position: 'insideBottom', offset: -2, fontSize: 12.5, fill: '#8a93a3' }} />
                <YAxis tick={tick} domain={[20, 100]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="wsABC" name="ABC · ws1000" stroke={C.teal} strokeWidth={2.6} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="wsSA" name="SA · ws1000" stroke={C.blue} strokeWidth={2.6} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="wsG" name="Greedy · ws1000" stroke={C.red} strokeWidth={2.2} strokeDasharray="5 4" dot={{ r: 3.5 }} />
                <Line type="monotone" dataKey="baABC" name="ABC · ba500" stroke={C.teal} strokeWidth={2} strokeDasharray="2 2" dot={false} />
                <Line type="monotone" dataKey="baSA" name="SA · ba500" stroke={C.blue} strokeWidth={2} strokeDasharray="2 2" dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <Legend items={[
              { label: 'ABC · ws1000', color: C.teal },
              { label: 'SA · ws1000', color: C.blue },
              { label: 'Greedy · ws1000', color: C.red },
              { label: 'ABC/SA · ba500 (≈ 94, both)', color: C.grey },
            ]} />
          </Panel>
        </div>
        <div className="col gap12" style={{ width: 440 }}>
          <Panel title="Ceiling checks">
            <div className="p" style={{ fontSize: 15.5 }}>
              ABC at 8× consumes <b>24,000/24,000</b> evaluations — flatness is a plateau.
              A 16× multi-start SA: 38.7 vs 37.6 at 4× — <b>no escape</b>. The sandwich
              bound is vacuous here (OPT ≤ 1495), so near-optimality rests on
              budget-insensitivity alone: 32× more budget buys &lt; 1 node.
            </div>
          </Panel>
          <Panel title="Wall-clock per configuration">
            <ResponsiveContainer width="100%" height={190}>
              <BarChart layout="vertical" data={wallClock} margin={{ top: 4, right: 16, left: 4, bottom: 0 }}>
                <CartesianGrid stroke={grid} horizontal={false} />
                <XAxis type="number" tick={tick} unit="s" />
                <YAxis type="category" dataKey="a" width={70} tick={{ fontSize: 12.5, fill: '#4a5468' }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v} s`, '']} />
                <Bar dataKey="s" radius={[0, 5, 5, 0]}>
                  {wallClock.map((r) => <Cell key={r.a} fill={r.a === 'ABC' ? C.teal : r.a === 'SA' ? C.blue : r.a.includes('CELF') ? C.orange : C.grey} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="small" style={{ marginTop: 6 }}>
              ABC is the cheapest set-level searcher. <b>Equal-wall-clock secondary:</b>{' '}
              ABC (≈2.4× budget) stays at 37.5 while SA at 1× reaches 30.7 — ABC wins
              the equal-time comparison.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/s9_budget.csv · s3_main.csv" n={16} i={10} />
  </div>
)

export const S12Delta = () => (
  <div className="slide">
    <Head
      kicker="11 · The δ-sweep — and its confound"
      title="Advantage vs threshold dispersion: real, then a caution"
      subtitle="The earlier “advantage tracks submodularity” claim does not survive control — the correlations are carried by δ itself."
      tag="§8.7"
    />
    <div className="body">
      <div className="split">
        <div className="col grow">
          <Panel title="ABC advantage over Greedy, held-out, vs δ">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={deltaWs} margin={{ top: 8, right: 14, left: -14, bottom: 2 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="d" tick={tick} label={{ value: 'δ', position: 'insideBottom', offset: -2, fontSize: 12.5, fill: '#8a93a3' }} />
                <YAxis tick={tick} unit="%" />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, '']} />
                <ReferenceLine y={0} stroke="#9aa3b3" />
                <Line type="monotone" dataKey="vsG" name="ABC vs Greedy · ws1000" stroke={C.teal} strokeWidth={2.6} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="small" style={{ marginTop: 6 }}>
              On ws1000 the advantage over Greedy is +24…+75% for δ ≤ 0.2, then gone —
              consistent with the plateau mechanism, not with the violation rate.
            </div>
          </Panel>
        </div>
        <div className="col gap12" style={{ width: 470 }}>
          <Panel title="ba500: the CELF gap, not the Greedy gap, tracks δ">
            <ResponsiveContainer width="100%" height={170}>
              <LineChart data={deltaBa} margin={{ top: 8, right: 14, left: -14, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="d" tick={tick} />
                <YAxis tick={tick} unit="%" />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, '']} />
                <Line type="monotone" dataKey="vsC" name="ABC vs CELF" stroke={C.orange} strokeWidth={2.4} dot={{ r: 3.5 }} />
                <Line type="monotone" dataKey="vsG" name="ABC vs Greedy" stroke={C.teal} strokeWidth={2.4} dot={{ r: 3.5 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="small">
              vs CELF: falls +54% → +0.4% as δ → 0.5 (Spearman −0.98, p = 3×10⁻⁵). vs Greedy:
              a wash at every δ (ρ = −0.29, p = 0.49).
            </div>
          </Panel>
          <div className="warn">
            <b>Confound, controlled:</b> partial correlation removing δ kills every
            association (p ≥ 0.31); a fixed-δ test across the 24 multi-instances is null
            (ρ = −0.23, p = 0.28). The structural claims rest on the proofs, the scans
            and the CELF audit — <b>not on this sweep</b>.
          </div>
        </div>
      </div>
    </div>
    <Foot src="results/s5_delta_ba500.csv · s5_delta_ws1000.csv" n={16} i={11} />
  </div>
)

export const S13Signed = () => (
  <div className="slide">
    <Head
      kicker="12 · Signed graphs"
      title="A real extension — and a control that says the advantage is the plateau"
      subtitle="Negative edges break monotonicity (witnessed). But the ABC-vs-Greedy advantage survives on the all-positive twin — it is not about the signs."
      tag="§9"
    />
    <div className="body">
      <div className="split">
        <div className="col gap12">
          <Panel title="Non-monotonicity — constructive witness">
            <div className="formula" style={{ fontSize: 16.5 }}>
              hub w: 4 pos leaves, 1 pos follower, 1 enemy leaf e<br />
              σ({'{1,2,3,4}'}) = <span className="hl">6</span>&nbsp;&nbsp;&gt;&nbsp;&nbsp;
              σ({'{1,2,3,4,5}'}) = <span className="red">5</span>
            </div>
            <div className="small" style={{ marginTop: 10 }}>
              Seeding the enemy leaf <b>blocks the hub</b>: a larger seed set with smaller
              influence. A drop-0.26 witness was also located on a random signed BA(200)
              graph by structured search. Greedy/CELF have no valid rationale on such
              objectives — ABC’s swaps can remove harmful seeds.
            </div>
          </Panel>
          <div className="kpis">
            <KPI tone="teal" v="17.6" l="ABC family mean (held-out)" />
            <KPI tone="blue" v="17.7" l="SA family mean" />
            <KPI v="15.4" l="Greedy" />
            <KPI tone="red" v="14.6" l="CELF" />
          </div>
        </div>
        <div className="col grow">
          <Panel title="Sign-ablation control — identical topology, signs stripped">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={signControl} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="g" tick={{ fontSize: 12.5, fill: '#4a5468' }} />
                <YAxis tick={tick} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="greedy" name="Greedy" fill={C.red} radius={[4, 4, 0, 0]} />
                <Bar dataKey="abc" name="ABC" fill={C.teal} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <Legend items={[{ label: 'Greedy', color: C.red }, { label: 'ABC', color: C.teal }]} />
            <div className="small" style={{ marginTop: 8 }}>
              sbm120 k = 15: <b>+95%</b> signed → <b>+181%</b> positive twin (+35.8 nodes vs
              +19.0). The signed contribution is the <b>model generalisation + the
              non-monotonicity witness</b> — not a distinct search advantage.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="results/s11_signcontrol.csv · s8_signed.csv" n={16} i={12} />
  </div>
)

export const S14Implementation = () => (
  <div className="slide">
    <Head
      kicker="13 · Implementation story"
      title="From the first implementation to the final submission"
      subtitle="What the original code did, where it fell short, which fixes worked, and which claims had to be abandoned."
      tag="§10 · corrections"
    />
    <div className="body">
      <div className="split-4">
        <Panel title="Original implementation">
          <div className="small">
            Started from the standard pipeline: Linear Threshold assumptions,
            Greedy + CELF, graph as an (n, k) row, budgets unaudited, no held-out
            scoring.
          </div>
          <div className="small" style={{ marginTop: 8 }}>
            <b>Shortcomings:</b> CELF’s lazy bound failed on 50% of re-checks;
            submodularity was false; small-graph wins vanished under equal budgets
            and fresh thresholds.
          </div>
        </Panel>
        <Panel title="Fixes and final optimisations">
          <div className="small">
            Replaced assumptions with proofs and audits; switched to equal-budget,
            held-out scoring with the <b>graph as the statistical unit</b>.
          </div>
          <div className="small" style={{ marginTop: 8 }}>
            <b>Optimiser:</b> canonical ABC → gbest-guided, tournament-onlooker,
            directed-scout, deduped, Φ-tie-break, CELF-seeded population search.
            Refinement was <b>removed</b>; ABC-CELF is kept only as an ablation.
          </div>
        </Panel>
        <Panel title="Where we failed — honestly">
          <div className="small">
            • Φ as a global fitness term <b>hurt</b>; Greedy+Φ trailed Greedy on WS.<br />
            • Metropolis acceptance helped (+2.5) but stayed below SA.<br />
            • The δ-sweep correlation was <b>confounded</b>; partial correlation
            nullifies it.<br />
            • “Signed graphs give ABC a distinct advantage” was rejected by the
            sign-ablation control.
          </div>
        </Panel>
        <Panel title="Final submission">
          <div className="small">
            <b>Package:</b> monotone proof + non-submodular witness + CELF audit +
            LOO-selected ABC + honest equal-budget evaluation (WS/BA/SBM) + signed
            extension.
          </div>
          <div className="small" style={{ marginTop: 8 }}>
            <b>Claim:</b> ABC is a budget-efficient, guarantee-carrying set-level
            searcher for a model where <b>coalitions carry the influence</b> — not the
            best possible searcher.
          </div>
        </Panel>
      </div>
    </div>
    <Foot src="MAJORITY_RULE_ABC_REPORT.md · §10–§11" n={16} i={13} />
  </div>
)

export const S16FutureScope = () => (
  <div className="slide">
    <Head
      kicker="15 · Future scope"
      title="Limitations and what comes next"
      subtitle="The current results are reproducible and Bonferroni-safe where stated; these are the remaining open doors."
      tag="§11 · limitations"
    />
    <div className="body">
      <div className="split-3">
        <Panel title="Current limitations">
          <div className="small">
            • Absolute effects are small: top methods differ by &lt;1% of n on
            average.<br />
            • BA and SBM evidence is exploratory (n = 8).<br />
            • Ceiling / near-optimality rests on budget-insensitivity; the sandwich
            bound is vacuous at this scale.<br />
            • Candidate pool is capped at 200 nodes on large graphs.
          </div>
        </Panel>
        <Panel title="Next steps">
          <div className="small">
            • Real-network signed tests: Epinions, Slashdot, Bitcoin-Alpha,
            ego-Facebook, email-Eu-core (<code>load_signed_edgelist</code> ready).<br />
            • Tighter ceiling certification: exact checks on larger subsamples or a
            non-vacuous upper bound.<br />
            • Budget-aware k selection: choose k relative to the estimated contagion
            threshold k*.<br />
            • Extend the audit to other non-submodular diffusion families.
          </div>
        </Panel>
        <Panel title="Final framing">
          <div className="p">
            The submission shows that for majority-rule diffusion on sparse,
            clustered graphs, <b>coalitions carry the influence</b> and a carefully
            audited set-level searcher is a sound, defensible choice.
          </div>
          <div className="chips" style={{ marginTop: 12 }}>
            <span className="chip on">reproducible code</span>
            <span className="chip on">honest negative results</span>
            <span className="chip">real networks next</span>
          </div>
        </Panel>
      </div>
    </div>
    <Foot src="MAJORITY_RULE_ABC_REPORT.md · §11" n={16} i={15} />
  </div>
)

export const S14Conclusions = () => (
  <div className="slide">
    <Head
      kicker="14 · Conclusions"
      title="Five claims that survive scrutiny — and three that don’t"
      tag="§12"
    />
    <div className="body">
      <div className="split">
        <div className="col gap10">
          <Panel title="What the evidence supports">
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="check">1.</span> The model is <b>monotone</b> (proved) and{' '}
              <b>not submodular</b> (witnessed, scanned, explained via Mossel–Roch).
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="check">2.</span> <b>CELF breaks</b>: 50.2% lazy violations;
              differs from Greedy in 16/17 configs, strictly worse in 13/17.
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="check">3.</span> On sparse clustered graphs, below the
              contagion threshold, <b>set-level search beats Greedy</b>: +4.6 held-out on
              16 WS instances (CI [+2.5, +6.9], Bonferroni-safe); ties it on BA; beats
              CELF everywhere.
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="check">4.</span> <b>ABC is a valid set-level searcher</b> —
              budget-efficient (ceiling at 1× vs SA’s 4×), ≥CELF guarantee, CV 1.7%.
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="check">5.</span> The <b>signed extension</b> adds the model
              and a non-monotonicity witness — no distinct search advantage.
            </div>
          </Panel>
        </div>
        <div className="col gap10">
          <Panel title="What not to claim">
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="cross">✗</span> “ABC is the best set-level searcher” —{' '}
              <b>SA ≥ ABC</b> on WS and BA.
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="cross">✗</span> “The advantage tracks submodularity” — the
              δ-sweep correlation is <b>confounded</b> and does not survive control.
            </div>
            <div className="p" style={{ fontSize: 15.5 }}>
              <span className="cross">✗</span> “Signed graphs need ABC specifically” — the
              <b> sign-ablation control</b> shows it is the plateau effect.
            </div>
            <div className="note" style={{ marginTop: 10 }}>
              Framing for submission: prove monotone; disprove submodularity and say why;
              quantify γ; audit CELF; present ABC as an effective, budget-efficient
              set-level optimiser for a model where <b>coalitions carry the influence</b>.
            </div>
          </Panel>
        </div>
      </div>
    </div>
    <Foot src="MAJORITY_RULE_ABC_REPORT.md · all results/*.csv" n={16} i={14} />
  </div>
)

export const SLIDES = [
  S1Title, S2Model, S3Structure, S4Scan, S5Celf, S6Abc, S7Protocol,
  S8Primary, S9Families, S10Regime, S11Budget, S12Delta, S13Signed,
  S14Implementation, S14Conclusions, S16FutureScope,
]