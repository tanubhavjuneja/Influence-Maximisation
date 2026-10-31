r"""
theory_checks.py
================
Computational verification of the structural properties of the majority rule
diffusion model.

The project brief asked us to "prove that the diffusion model is submodular and
monotone".  Half of that is provable.  The other half is provably FALSE, and
this module is the machine-checked evidence.

  PROPOSITION 1 (monotonicity).  sigma is monotone non-decreasing.
      PROOF.  Fix a threshold realisation theta.  Let S subset T.  Run the two
      cascades in lock-step and induct on time t.  Claim: A_t(S) subset A_t(T).
      Base t=0: A_0(S)=S subset T=A_0(T).  Step: if v in A_{t+1}(S)\A_t(S) then
      |N(v) cap A_t(S)| > theta_v deg(v); by the inductive hypothesis
      A_t(S) subset A_t(T) so |N(v) cap A_t(T)| > theta_v deg(v), hence
      v in A_{t+1}(T).  Progressivity means nothing is ever removed.  Therefore
      |A_inf(S,theta)| <= |A_inf(T,theta)| pointwise, and taking expectations
      over theta preserves the inequality.  QED
      -> verified empirically by `check_monotonicity`.

  PROPOSITION 2 (non-submodularity).  For every genuine majority rule
      studied here sigma is NOT submodular -- established computationally
      (machine-checked counterexamples below), with the failure of the
      Mossel-Roch sufficient condition as the structural explanation.
      ARGUMENT.  Mossel & Roch (2007) settled the Kempe-Kleinberg-Tardos
      conjecture: under the general threshold model, IF every local activation
      function
              f_v(S) = Pr[ theta_v < |S cap N(v)| / deg(v) ]
                     = F_theta( |S cap N(v)| / deg(v) )
      is submodular in S THEN sigma is submodular (submodularity of the local
      functions is a SUFFICIENT condition -- Mossel & Roch do not claim the
      converse).  Since f_v depends on S only through the count
      a = |S cap N(v)|, submodularity of f_v is equivalent to CONCAVITY of the
      threshold CDF F_theta on the grid {0, 1/d, ..., 1}.
        * theta ~ U[0,1]  =>  F(x) = x, linear => concave => LT is submodular.
        * theta concentrated near 1/2 => F is an S-curve: flat, then rising,
          then flat.  The initial flat-then-rising stretch is strictly CONVEX,
          so F is not concave and f_v is not submodular.
        * theta = 1/2 a.s. => F is a step function, the extreme case.
      Concavity of F is precisely the statement "each extra active neighbour
      helps less than the previous one".  A majority rule says the opposite
      below the threshold -- the LAST neighbour needed is worth everything and
      the ones before it are worth nothing.  Diminishing returns is structurally
      incompatible with a majority requirement.
      The failure of the Mossel-Roch sufficient condition does NOT by itself
      prove sigma non-submodular, so the actual violations are established
      computationally: the counterexamples below are machine-checked
      witnesses -- `minimal_counterexample` and `exhaustive_violation_scan`.

  COROLLARY.  The (1 - 1/e) greedy guarantee does not apply.  Hardness: for
      the general deterministic threshold model (arbitrary thresholds)
      Lu, Zhang, Wu, Kim & Fu (J. Comb. Optim. 24(3):374-378, 2012) show
      there is no polynomial-time n^(1-eps) approximation unless P = NP.
      Our uniform-1/2 majority rule is a special case of their setting, so
      their inapproximability does NOT automatically transfer to it -- the
      approximation behaviour we report is measured empirically against the
      brute-force optimum instead.

  PROPOSITION 3 (what we CAN salvage).  Das & Kempe's submodularity ratio
              gamma_{U,k} = min_{L, S:|S|<=k}
                        sum_{v in S}[f(L+v)-f(L)] / [f(L u S) - f(L)]
      still yields a (1 - e^{-gamma}) guarantee for forward greedy.  We
      estimate gamma empirically -- it is the honest, instance-specific
      replacement for the textbook 0.632 factor.
      -> `submodularity_ratio`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np

from majority_rule_model import MajorityRuleModel, candidate_pool


# --------------------------------------------------------------------------- #
#  Proposition 1 - monotonicity
# --------------------------------------------------------------------------- #
def check_monotonicity(model: MajorityRuleModel, trials: int = 2000, seed: int = 0) -> dict:
    """Sample S subset T and confirm sigma(S) <= sigma(T) every single time."""
    rng = np.random.default_rng(seed)
    violations, worst = 0, 0.0
    for _ in range(trials):
        size_t = int(rng.integers(1, min(8, model.n) + 1))
        T = rng.choice(model.n, size=size_t, replace=False)
        size_s = int(rng.integers(0, size_t))
        S = T[:size_s]
        a, b = model.sigma(S), model.sigma(T)
        if a > b + 1e-9:
            violations += 1
            worst = max(worst, a - b)
    return {
        "property": "monotonicity",
        "trials": trials,
        "violations": violations,
        "worst_gap": worst,
        "holds": violations == 0,
    }


# --------------------------------------------------------------------------- #
#  Proposition 2 - the minimal counterexample
# --------------------------------------------------------------------------- #
def minimal_counterexample(verbose: bool = True) -> dict:
    """
    The smallest graph on which strict-majority diffusion violates submodularity.

        w is adjacent to x, y, z   (deg(w) = 3, so w needs 2 active neighbours)
        x, y, z have no other edges

        S = {}          T = {x}          added element v = y

        sigma(S + y) - sigma(S) = 1 - 0 = 1        (y alone cannot flip w)
        sigma(T + y) - sigma(T) = 4 - 1 = 3        (x, y flip w, w flips z)

        3 > 1  =>  INCREASING returns  =>  submodularity fails.

    Intuition: y is a *swing voter*.  Its value is zero until the rest of the
    coalition is almost large enough, and then it is worth the whole node.
    """
    G = nx.Graph()
    G.add_edges_from([("w", "x"), ("w", "y"), ("w", "z")])
    m = MajorityRuleModel(G, variant="dmr", n_samples=1, seed=0)
    i = m.index

    S: list[int] = []
    T = [i["x"]]
    v = i["y"]

    gS = m.sigma(S + [v]) - m.sigma(S)
    gT = m.sigma(T + [v]) - m.sigma(T)

    res = {
        "property": "submodularity",
        "graph": "star K_1,3 (w centre)",
        "S": "{}",
        "T": "{x}",
        "v": "y",
        "gain_S": gS,
        "gain_T": gT,
        "submodular": gS >= gT - 1e-9,
        "violation_magnitude": gT - gS,
    }
    if verbose:
        print("--- minimal counterexample (deterministic majority) ---")
        print(f"  sigma(S+v) - sigma(S) = {gS}")
        print(f"  sigma(T+v) - sigma(T) = {gT}")
        print(f"  submodular? {res['submodular']}  (need gain_S >= gain_T)")
    return res


def star_counterexample_sweep(n_leaves: int = 3, t_size: int = 1,
                              deltas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
                              R: int = 20000, seed: int = 0) -> list[dict]:
    """
    Does softening the threshold rescue submodularity?  Re-run the star
    counterexample with theta ~ U[1/2-delta, 1/2+delta] on a K_{1,n} star,
    S = {}, T = the first t_size leaves, v = the next leaf.

    NOTE the K_1,3 star only violates clearly for delta below about 1/3;
    larger stars keep the violation visible up to delta ~ 0.45.  delta = 0.5
    is exactly the Linear Threshold model, where submodularity holds, so it
    acts as a positive control.
    """
    G = nx.Graph()
    G.add_edges_from((0, i) for i in range(1, n_leaves + 1))
    out = []
    for d in deltas:
        variant = "dmr" if d == 0.0 else ("lt" if d == 0.5 else "smr")
        m = MajorityRuleModel(G, variant=variant, delta=d, n_samples=R, seed=seed)
        T = list(range(1, t_size + 1))
        v = t_size + 1
        gS = m.sigma([v]) - m.sigma([])
        gT = m.sigma(T + [v]) - m.sigma(T)
        out.append({
            "star": f"K_1,{n_leaves}", "T": T, "v": v,
            "delta": d,
            "variant": variant,
            "gain_S": gS,
            "gain_T": gT,
            "violation": gT - gS,
            "submodular": gS >= gT - 1e-3,
        })
    return out


def counterexample_survives_stochasticity(deltas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
                                          R: int = 20000, seed: int = 0) -> list[dict]:
    """Backwards-compatible wrapper: the minimal K_1,3 counterexample sweep."""
    return star_counterexample_sweep(3, 1, deltas=deltas, R=R, seed=seed)


def exhaustive_violation_scan(
    model: MajorityRuleModel,
    max_T: int = 3,
    node_cap: int = 30,
    seed: int = 0,
    tol: float | None = None,
) -> dict:
    r"""
    Brute-force every (S, T, v) with S subset T, |T| <= max_T, v not in T over a
    capped node set, and count how many violate the diminishing-returns
    inequality  gain(S, v) >= gain(T, v).

    Two subtleties, both of which matter for the honesty of the result:

    (a) MONTE-CARLO NOISE.  sigma is estimated from R threshold draws, so a
        strictly submodular function can still show apparent violations of
        order the sampling error.  We therefore require the violation to exceed
        `tol`, which defaults to a conservative multiple of the Monte-Carlo
        standard error of sigma.  With this filter the Linear Threshold control
        condition correctly reports (near-)zero violations.
        NOTE: for a *fixed* threshold vector the threshold process is not
        submodular -- LT's submodularity is a property of the expectation only.
        This is exactly why the tolerance is needed.

    (b) DEGENERATE TRIPLES.  Under a strict majority rule most small seed sets
        activate nobody, so gain(S,v) = gain(T,v) = 1 (the node itself) and the
        inequality holds vacuously.  Counting those dilutes the statistic, so we
        additionally report the violation rate restricted to triples where at
        least one of the two gains exceeds 1, i.e. where diffusion actually
        happened.
    """
    rng = np.random.default_rng(seed)
    nodes = list(range(model.n))
    if model.n > node_cap:
        deg_order = sorted(nodes, key=lambda i: -model.deg[i])
        nodes = sorted(set(deg_order[: node_cap // 2] +
                           rng.choice(nodes, node_cap - node_cap // 2,
                                      replace=False).tolist()))

    if tol is None:
        probe = [model.sigma_std(rng.choice(nodes, 3, replace=False)) for _ in range(12)]
        tol = 4.0 * float(np.mean(probe)) + 1e-9

    total = viol = 0
    nd_total = nd_viol = 0
    worst = 0.0
    worst_witness = None
    for t_size in range(1, max_T + 1):
        for T in itertools.combinations(nodes, t_size):
            Tset = set(T)
            rest = [v for v in nodes if v not in Tset]
            for s_size in range(0, t_size):
                for S in itertools.combinations(T, s_size):
                    for v in rest:
                        gS = model.sigma(set(S) | {v}) - model.sigma(set(S))
                        gT = model.sigma(Tset | {v}) - model.sigma(Tset)
                        total += 1
                        active = max(gS, gT) > 1.0 + tol
                        nd_total += active
                        if gT > gS + tol:
                            viol += 1
                            nd_viol += active
                            if gT - gS > worst:
                                worst = gT - gS
                                worst_witness = (tuple(S), tuple(T), v, gS, gT)
    return {
        "property": "submodularity (exhaustive scan)",
        "tolerance": tol,
        "triples_tested": total,
        "violations": viol,
        "violation_rate": viol / total if total else 0.0,
        "nondegenerate_triples": nd_total,
        "nondegenerate_violation_rate": nd_viol / nd_total if nd_total else 0.0,
        "worst_violation": worst,
        "worst_witness": worst_witness,
        "submodular": viol == 0,
    }


# --------------------------------------------------------------------------- #
#  Local threshold function: the Mossel-Roch view
# --------------------------------------------------------------------------- #
def local_function_curvature(variant: str = "smr", delta: float = 0.15,
                             degree: int = 10) -> dict:
    """
    Tabulate f_v(a) = Pr[theta < a/d] for a = 0..d and report the second
    differences.  Submodularity of f_v <=> all second differences <= 0
    (discrete concavity).  Any positive second difference shows the
    Mossel-Roch sufficient condition FAILS for that v -- it motivates the
    empirical scan, which is what actually certifies sigma's violations.
    """
    a = np.arange(degree + 1)
    x = a / degree
    if variant == "lt":
        F = np.clip(x, 0, 1)
    elif variant == "dmr":
        F = (x > 0.5).astype(float)
    else:
        lo, hi = 0.5 - delta, 0.5 + delta
        F = np.clip((x - lo) / (hi - lo), 0, 1)
    d1 = np.diff(F)
    d2 = np.diff(d1)
    return {
        "variant": variant,
        "delta": delta,
        "degree": degree,
        "F": F.tolist(),
        "first_differences": d1.tolist(),
        "second_differences": d2.tolist(),
        "max_second_difference": float(d2.max()),
        "concave_hence_submodular": bool(d2.max() <= 1e-12),
    }


# --------------------------------------------------------------------------- #
#  Proposition 3 - submodularity ratio (Das & Kempe 2011)
# --------------------------------------------------------------------------- #
@dataclass
class RatioEstimate:
    gamma_min: float
    gamma_mean: float
    gamma_p05: float
    samples: int
    greedy_bound: float          # 1 - e^{-gamma_min}
    classic_bound: float = 1 - np.exp(-1.0)

    def as_dict(self) -> dict:
        return {
            "gamma_min": self.gamma_min,
            "gamma_mean": self.gamma_mean,
            "gamma_p05": self.gamma_p05,
            "samples": self.samples,
            "greedy_bound_1_minus_exp_neg_gamma": self.greedy_bound,
            "classic_bound_1_minus_1_over_e": self.classic_bound,
        }


def submodularity_ratio(
    model: MajorityRuleModel,
    k: int = 5,
    pool: list[int] | None = None,
    samples: int = 400,
    seed: int = 0,
) -> RatioEstimate:
    """
    Monte-Carlo estimate of gamma_{U,k}.

    gamma <= 1 always; gamma = 1 exactly for submodular f.  The greedy
    algorithm on a monotone function with ratio gamma achieves
    (1 - e^{-gamma}) * OPT, so gamma is the instance-specific replacement for
    the 0.632 constant.
    """
    rng = np.random.default_rng(seed)
    U = pool if pool is not None else list(range(model.n))
    ratios: list[float] = []
    for _ in range(samples):
        l_size = int(rng.integers(0, k + 1))
        L = set(rng.choice(U, size=l_size, replace=False).tolist())
        s_size = int(rng.integers(1, k + 1))
        S = set(rng.choice([u for u in U if u not in L], size=s_size, replace=False).tolist())
        fL = model.sigma(L)
        denom = model.sigma(L | S) - fL
        # noise floor: ignore pairs whose joint gain is within sampling error of
        # zero, otherwise the ratio is dominated by Monte-Carlo noise
        if denom <= max(1.0, 4.0 * model.sigma_std(L | S)):
            continue
        numer = sum(model.sigma(L | {v}) - fL for v in S)
        ratios.append(numer / denom)
    arr = np.array(ratios) if ratios else np.array([1.0])
    gmin = float(arr.min())
    return RatioEstimate(
        gamma_min=gmin,
        gamma_mean=float(arr.mean()),
        gamma_p05=float(np.percentile(arr, 5)),
        samples=len(arr),
        greedy_bound=float(1 - np.exp(-gmin)),
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from majority_rule_model import load_graph

    print("=" * 70)
    minimal_counterexample()

    print("\n--- does softening the threshold restore submodularity? (K_1,3) ---")
    for row in counterexample_survives_stochasticity():
        flag = "OK " if row["submodular"] else "FAIL"
        print(f"  delta={row['delta']:.2f} [{row['variant']:3s}] "
              f"gain_S={row['gain_S']:.4f} gain_T={row['gain_T']:.4f} "
              f"violation={row['violation']:+.4f}  {flag}")

    print("\n--- same sweep on a larger star (K_1,7, T = 3 leaves) ---")
    for row in star_counterexample_sweep(7, 3, deltas=(0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.49, 0.5)):
        flag = "OK " if row["submodular"] else "FAIL"
        print(f"  delta={row['delta']:.2f} [{row['variant']:3s}] "
              f"gain_S={row['gain_S']:.4f} gain_T={row['gain_T']:.4f} "
              f"violation={row['violation']:+.4f}  {flag}")

    print("\n--- local activation function (Mossel-Roch test, d=10) ---")
    for v, d in (("lt", 0.5), ("smr", 0.15), ("dmr", 0.0)):
        r = local_function_curvature(v, d, 10)
        print(f"  {v:4s} max 2nd-difference = {r['max_second_difference']:+.4f} "
              f"-> submodular: {r['concave_hence_submodular']}")

    G = load_graph("karate")
    m = MajorityRuleModel(G, "smr", 0.15, 200, seed=1)
    print(f"\n{m.describe()}")
    print("\n--- monotonicity ---")
    print(" ", check_monotonicity(m, 1000))
    print("\n--- exhaustive submodularity scan ---")
    print(" ", exhaustive_violation_scan(m, max_T=3, node_cap=18))
    print("\n--- submodularity ratio ---")
    print(" ", submodularity_ratio(m, k=5, samples=300).as_dict())
