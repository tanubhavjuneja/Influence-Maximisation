"""
graphio.py
==========
SNAP signed-network parsing with the STANDARD LIBRARY only (gzip, urllib)
plus numpy for the arrays.  No networkx.

Datasets
--------
  bitcoin_alpha : soc-sign-bitcoinalpha.csv.gz   source,target,rating,time
  bitcoin_otc   : soc-sign-bitcoinotc.csv.gz
  slashdot      : soc-sign-Slashdot090221.txt.gz  source target sign  ('#' comments)

Processing (same convention as the main project)
-----------------------------------------------
  * drop self-loops and zero ratings,
  * UNDIRECTED: each pair kept once, multi-ratings collapsed to the max |w|
    (the sign of that kept rating),
  * nodes relabelled to 0..n-1 (sorted original ids).

Returns a dict: name, n, m, u, v (int64 arrays), s (sign, int8),
w (|rating|, int32).  Slashdot weights are all 1.
"""
import gzip
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SOURCES = {
    "bitcoin_alpha": ("https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
                      "soc-sign-bitcoinalpha.csv.gz"),
    "bitcoin_otc": ("https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
                    "soc-sign-bitcoinotc.csv.gz"),
    "slashdot": ("https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz",
                 "soc-sign-Slashdot090221.txt.gz"),
}


def find_raw(name: str) -> Path:
    """Local file first (signed_bench/data, then ../data), else download."""
    fn = SOURCES[name][1]
    for p in (DATA / fn, HERE.parent / "data" / fn):
        if p.exists():
            return p
    DATA.mkdir(exist_ok=True)
    url = SOURCES[name][0]
    print(f"  downloading {name} <- {url}")
    dest = DATA / fn
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _parse_bitcoin(path: Path):
    us, vs, ws = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            try:
                w = int(float(p[2]))
            except ValueError:
                continue
            if w == 0:
                continue
            u, v = int(p[0]), int(p[1])
            if u == v:
                continue
            us.append(u)
            vs.append(v)
            ws.append(w)
    return (np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64),
            np.asarray(ws, dtype=np.int32))


def _parse_slashdot(path: Path):
    us, vs, ws = [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            w = int(p[2])
            if w == 0:
                continue
            u, v = int(p[0]), int(p[1])
            if u == v:
                continue
            us.append(u)
            vs.append(v)
            ws.append(w)
    return (np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64),
            np.asarray(ws, dtype=np.int32))


def load_graph(name: str, verbose: bool = True) -> dict:
    path = find_raw(name)
    if name.startswith("bitcoin"):
        u, v, w = _parse_bitcoin(path)
    else:
        u, v, w = _parse_slashdot(path)

    # relabel to 0..n-1 (sorted original ids)
    all_nodes = np.unique(np.concatenate([u, v]))
    mapping = {int(x): i for i, x in enumerate(all_nodes)}
    u = np.asarray([mapping[int(x)] for x in u], dtype=np.int64)
    v = np.asarray([mapping[int(x)] for x in v], dtype=np.int64)
    n = int(len(all_nodes))

    # undirectedise + collapse multi-edges, keeping max |w| (its sign)
    best: dict[int, int] = {}
    for i in range(len(u)):
        a, b = int(u[i]), int(v[i])
        if a > b:
            a, b = b, a
        key = a * n + b
        ww = int(w[i])
        if key not in best or abs(ww) > abs(best[key]):
            best[key] = ww

    uu = np.fromiter((k // n for k in best), dtype=np.int64, count=len(best))
    vv = np.fromiter((k % n for k in best), dtype=np.int64, count=len(best))
    ww = np.fromiter(best.values(), dtype=np.int32, count=len(best))
    s = np.where(ww > 0, np.int8(1), np.int8(-1))
    w = np.abs(ww).astype(np.int32)
    G = {"name": name, "n": n, "m": int(len(uu)),
         "u": uu, "v": vv, "s": s, "w": w}
    if verbose:
        print(f"  {name}: n={n} m={len(uu)} "
              f"pos={int((s > 0).sum())} neg={int((s < 0).sum())}")
    return G


def directed_pairs(G: dict):
    """Both directions of every edge with signed weight (len 2m)."""
    src = np.concatenate([G["u"], G["v"]])
    dst = np.concatenate([G["v"], G["u"]])
    sw = np.concatenate([(G["s"] * G["w"]).astype(np.float32),
                         (G["s"] * G["w"]).astype(np.float32)])
    return src.astype(np.int64), dst.astype(np.int64), sw.astype(np.float32)


if __name__ == "__main__":
    for g in ("bitcoin_alpha", "bitcoin_otc"):
        load_graph(g)