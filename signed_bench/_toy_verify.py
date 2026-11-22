"""Verify the 9-node demo graph used in the report."""
import itertools

import numpy as np

EDGES = [
    (3, 5, 1), (3, 7, 1), (6, 8, 1), (3, 8, 1), (5, 8, 1),
    (1, 9, 1), (3, 4, 1), (2, 8, 1), (4, 5, 1), (6, 7, 1),
    (2, 9, -1), (1, 8, -1), (1, 7, -1), (1, 3, -1), (1, 6, -1),
]
N = 9
THETA = 0.5


def build():
    adj = [[] for _ in range(N + 1)]
    for u, v, s in EDGES:
        adj[u].append((v, s))
        adj[v].append((u, s))
    return adj


def simulate(seeds, trace=False):
    adj = build()
    deg = {v: len(adj[v]) for v in range(1, N + 1)}
    need = {v: int(np.floor(THETA * deg[v])) + 1 for v in deg}
    state = {v: 0 for v in range(1, N + 1)}
    for v in seeds:
        state[v] = 1
    if trace:
        print(f"seeds {seeds}: need={need}")
        print(f"  round 0: {state}")
    rnd = 0
    while True:
        support = {v: 0 for v in range(1, N + 1)}
        for u in range(1, N + 1):
            if state[u] != 0:
                for v, s in adj[u]:
                    support[v] += s * state[u]
        new = [v for v in range(1, N + 1)
               if state[v] == 0 and abs(support[v]) >= need[v]]
        if not new:
            break
        rnd += 1
        for v in new:
            state[v] = int(np.sign(support[v]))
        if trace:
            print(f"  round {rnd}: support={support}")
            print(f"           active={ {v: state[v] for v in new} }")
    pos = sum(1 for v in state.values() if v == 1)
    neg = sum(1 for v in state.values() if v == -1)
    return state, pos, neg


def main():
    st, pos, neg = simulate([3, 6], trace=True)
    print(f"final {st} pos={pos} neg={neg} net={pos-neg}\n")

    print("singletons:")
    for v in range(1, N + 1):
        st, pos, neg = simulate([v])
        print(f"  {{{v}}}: pos={pos} neg={neg} net={pos-neg}")

    print("\npairs (top by net):")
    rows = []
    for S in itertools.combinations(range(1, N + 1), 2):
        st, pos, neg = simulate(list(S))
        rows.append((pos - neg, pos, neg, S))
    rows.sort(reverse=True)
    for r in rows[:5]:
        print(" ", r)

    print("\nmonotonicity violations (adding one seed lowers net):")
    found = 0
    for size in (1, 2, 3):
        for S in itertools.combinations(range(1, N + 1), size):
            _, p0, n0 = simulate(list(S))
            base = p0 - n0
            for v in range(1, N + 1):
                if v in S:
                    continue
                _, p1, n1 = simulate(list(S) + [v])
                if p1 - n1 < base:
                    print(f"  S={S} net={base}  +{v} -> net={p1-n1} "
                          f"(pos {p1}, neg {n1})")
                    found += 1
                    if found >= 3:
                        return


if __name__ == "__main__":
    main()
