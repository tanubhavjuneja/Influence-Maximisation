import csv
import statistics
from collections import defaultdict

rows = list(csv.DictReader(open("signed_bench/results.csv", encoding="utf-8")))
print("cells:", len(rows))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r["graph"], r["algorithm"], int(r["k"]))
    agg[key]["t"].append(float(r["time_s"]))
    agg[key]["p"].append(float(r["positive_spread"]))
    agg[key]["n"].append(float(r["negative_spread"]))
    agg[key]["tot"].append(float(r["total_spread"]))
    agg[key]["f"].append(float(r["fitness"]))

for g in ["bitcoin_alpha", "bitcoin_otc"]:
    print()
    print("=" * 8, g)
    hdr = f"{'alg':18s} {'k':>4s} {'time_s':>7s} {'pos':>8s} {'neg':>8s} " \
          f"{'tot':>8s} {'fitness':>10s}  runs"
    print(hdr)
    for alg in ["Random", "Degree", "ABC (proposed)", "ABCVNS (paper)"]:
        for k in [10, 20, 30, 40, 50]:
            d = agg[(g, alg, k)]
            n_runs = len(d["t"])
            print(f"{alg:18s} {k:4d} {statistics.mean(d['t']):7.1f} "
                  f"{statistics.mean(d['p']):8.1f} {statistics.mean(d['n']):8.1f} "
                  f"{statistics.mean(d['tot']):8.1f} {statistics.mean(d['f']):10.1f}  {n_runs}")