r"""
real_datasets.py
================
Loaders + local cache for the three open SNAP networks used in the
real-network comparison (supervisor task):

  * Bitcoin Alpha web-of-trust  (soc-sign-bitcoinalpha.csv.gz)
  * Bitcoin OTC   web-of-trust  (soc-sign-bitcoinotc.csv.gz)
  * Slashdot Zoo signed network (soc-sign-Slashdot090221.txt.gz)

All three are signed, directed, open-source datasets (SNAP, https://snap.stanford.edu).

Every loader:
  * relabels nodes to 0..n-1,
  * drops self-loops,
  * collapses multi-edges between a pair (keeps the max |rating|),
  * returns one UNDIRECTED networkx.Graph whose edges carry `sign`
    (in {+1,-1}) and `weight` (= abs rating, always 1 for Slashdot).
    The UNSIGNED majority / IC models simply ignore the attributes;
    the SIGNED model uses them.

Parsing is cached (npz + gpickle + stats.json), so re-runs skip re-parsing:
this is the first "save as you generate" layer for the big graphs.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

DATA = Path("data")
DATA.mkdir(exist_ok=True)

URLS = {
    "bitcoin_alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "bitcoin_otc": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    "slashdot": "https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz",
}
FILES = {
    "bitcoin_alpha": DATA / "soc-sign-bitcoin-alpha.csv.gz",
    "bitcoin_otc": DATA / "soc-sign-bitcoin-otc.csv.gz",
    "slashdot": DATA / "soc-sign-Slashdot090221.txt.gz",
}

BIG_COMPONENT = {
    # relabel internal node id of a node in the giant component, so
    # slashdot metrics (BC approx etc.) run on the component people
    # actually live in; cached stats also report component sizes.
    "bitcoin_alpha": None,
    "bitcoin_otc": None,
    "slashdot": None,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_datasets(force: bool = False) -> dict[str, Path]:
    """Download any missing datasets (idempotent)."""
    import urllib.request

    for name, url in URLS.items():
        path = FILES[name]
        if path.exists() and not force:
            continue
        log(f"  downloading {name} <- {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
        log(f"    saved {path} ({path.stat().st_size/1e6:.1f} MB)")
    return FILES


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #
def _parse_bitcoin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, w) int arrays with w = rating in [-10, 10]."""
    us, vs, ws = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            u, v = int(parts[0]), int(parts[1])
            try:
                w = int(float(parts[2]))
            except ValueError:
                continue
            if u == v or w == 0:
                continue
            us.append(u), vs.append(v), ws.append(w)
    return (np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64),
            np.asarray(ws, dtype=np.int32))


def _parse_slashdot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, w) with w in {-1, +1}; skips the 4 comment lines."""
    us, vs, ws = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, v, w = int(parts[0]), int(parts[1]), int(parts[2])
            if u == v or w == 0:
                continue
            us.append(u), vs.append(v), ws.append(w)
    return (np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64),
            np.asarray(ws, dtype=np.int32))


def _collapse_edges(u, v, w):
    """Undirectedise + collapse multi-edges, keeping max |w| (sign of that w)."""
    best: dict[tuple[int, int], int] = {}
    for a, b, ww in zip(u.tolist(), v.tolist(), w.tolist()):
        key = (a, b) if a <= b else (b, a)
        if key not in best or abs(ww) > abs(best[key]):
            best[key] = ww
    uu = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
    vv = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
    ww = np.fromiter(best.values(), dtype=np.int32, count=len(best))
    return uu, vv, ww


# --------------------------------------------------------------------------- #
#  Loader (cached)
# --------------------------------------------------------------------------- #
def load_real_graph(name: str, use_cache: bool = True, verbose: bool = True
                    ) -> nx.Graph:
    """
    Load one of bitcoin_alpha / bitcoin_otc / slashdot as an undirected
    graph with edge attributes sign, weight.  Cached to
      data/cache_<name>.npz   (edge arrays)
      data/cache_<name>.gpickle
      data/cache_<name>_stats.json
    """
    path = FILES[name]
    if not path.exists():
        ensure_datasets()
    npz = DATA / f"cache_{name}.npz"
    pkl = DATA / f"cache_{name}.pkl"
    stats_f = DATA / f"cache_{name}_stats.json"

    if use_cache and npz.exists() and pkl.exists():
        import pickle
        t0 = time.perf_counter()
        with open(pkl, "rb") as f:
            G = pickle.load(f)
        if verbose:
            log(f"  {name}: loaded from cache ({time.perf_counter()-t0:.1f}s, "
                f"n={G.number_of_nodes()}, m={G.number_of_edges()})")
        return G

    t0 = time.perf_counter()
    if verbose:
        log(f"  {name}: parsing {path.name} ...")
    if name.startswith("bitcoin"):
        u, v, w = _parse_bitcoin(path)
    else:
        u, v, w = _parse_slashdot(path)

    # relabel to 0..n-1
    all_nodes = np.unique(np.concatenate([u, v]))
    mapping = {int(x): i for i, x in enumerate(all_nodes)}
    u = np.asarray([mapping[int(x)] for x in u], dtype=np.int64)
    v = np.asarray([mapping[int(x)] for x in v], dtype=np.int64)

    u, v, w = _collapse_edges(u, v, w)
    n = int(len(all_nodes))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, ww in zip(u.tolist(), v.tolist(), w.tolist()):
        G.add_edge(a, b, sign=1 if ww > 0 else -1, weight=abs(int(ww)))

    pos = (w > 0).sum()
    neg = (w < 0).sum()

    if use_cache:
        import pickle
        np.savez_compressed(npz, u=u, v=v, w=w)
        with open(pkl, "wb") as f:
            pickle.dump(G, f, protocol=5)
        stats = {
            "name": name, "n": n, "m": int(len(u)),
            "m_pos": int(pos), "m_neg": int(neg),
            "max_deg": int(max(dict(G.degree()).values())),
            "avg_deg": float(np.mean(list(dict(G.degree()).values()))),
            "n_components": nx.number_connected_components(G),
        }
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        stats["giant_size"] = len(comps[0]) if comps else 0
        stats["giant_fraction"] = stats["giant_size"] / n if n else 0.0
        stats_f.write_text(json.dumps(stats, indent=2))
        if verbose:
            log(f"  {name}: parsed + cached in {time.perf_counter()-t0:.1f}s")
    if verbose:
        log(f"  {name}: n={n} m={len(u)} pos={pos} neg={neg}")
    return G


def dataset_stats(name: str) -> dict:
    stats_f = DATA / f"cache_{name}_stats.json"
    if stats_f.exists():
        return json.loads(stats_f.read_text())
    return {}


def load_outdegree(name: str, use_cache: bool = True) -> np.ndarray:
    """Out-degree in the ORIGINAL directed graph, aligned to 0..n-1.

    The undirected cache drops edge direction, so the raw file is parsed
    once more and outgoing edges are counted per relabelled node (the
    relabelling is the same deterministic sorted mapping used by
    `load_real_graph`).  Cached to `data/outdeg_<name>.npy`.
    """
    out_f = DATA / f"outdeg_{name}.npy"
    if use_cache and out_f.exists():
        return np.load(out_f)
    path = FILES[name]
    if not path.exists():
        ensure_datasets()
    if name.startswith("bitcoin"):
        u, v, w = _parse_bitcoin(path)
    else:
        u, v, w = _parse_slashdot(path)
    all_nodes = np.unique(np.concatenate([u, v]))
    mapping = {int(x): i for i, x in enumerate(all_nodes)}
    out = np.zeros(len(all_nodes), dtype=np.int64)
    for a in u.tolist():
        out[mapping[int(a)]] += 1
    np.save(out_f, out)
    log(f"  {name}: out-degree cached ({out_f.name}, "
        f"max={int(out.max())}, mean={out.mean():.2f})")
    return out


if __name__ == "__main__":
    ensure_datasets()
    for g in ("bitcoin_alpha", "bitcoin_otc", "slashdot"):
        G = load_real_graph(g)
        print(dataset_stats(g))