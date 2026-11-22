"""Search for a 9-node, 15-edge (5 negative) demo graph whose two-opinion
cascade is interesting: a small coalition flips a cluster while every
singleton does nothing, and at least one node ends negative."""
import itertools
import random

import numpy as np

N, M, NEG = 9, 15, 5
THETA = 0.5


def need(deg):
    return int(np.floor(THETA * deg)) + 1


def simulate(edges, seeds):
    adj = [[] for _ in range(N)]
    for u, v, s in edges:
        adj[u].append((v, s))
        adj[v].append((u, s))
    deg = np.array([len(a) for a in adj])
    nd = np.array([need(d) if d else 10**9 for d in deg])
    state = np.zeros(N, dtype=int)
    for v in seeds:
        state[v] = 1
    rounds = [state.copy()]
    while True:
        support = np.zeros(N, dtype=int)
        for u in range(N):
            if state[u] != 0:
                for v, s in adj[u]:
                    support[v] += s * state[u]
        new = (np.abs(support) >= nd) & (state == 0)
        if not new.any():
            break
        state[new] = np.sign(support[new]).astype(int)
        rounds.append(state.copy())
    return state, rounds, deg, nd


def score(edges, k=2):
    best = None
    for seeds in itertools.combinations(range(N), k):
        st, rounds, deg, nd = simulate(edges, seeds)
        pos = int((st == 1).sum())
        neg = int((st == -1).sum())
        if best is None or (pos - neg) > best[0]:
            best = (pos - neg, pos, neg, seeds, st, rounds)
    # singleton max
    single = 0
    for v in range(N):
        st, *_ = simulate(edges, [v])
        single = max(single, int((st == 1).sum()))
    return best, single


def main():
    rng = random.Random(7)
    all_edges = list(itertools.combinations(range(N), 2))
    found = 0
    for trial in range(200000):
        chosen = rng.sample(all_edges, M)
        negs = set(rng.sample(chosen, NEG))
        edges = [(u, v, -1 if (u, v) in negs else 1) for (u, v) in chosen]
        best, single = score(edges, k=2)
        if best and best[1] >= 6 and best[2] >= 1 and single == 1:
            print("FOUND trial", trial)
            print("edges:", edges)
            print("best 2-seed:", best[:4])
            print("final:", best[4].tolist())
            print("rounds:", [r.tolist() for r in best[5]])
            st, rounds, deg, nd = simulate(edges, best[3])
            print("deg:", deg.tolist(), "need:", nd.tolist())
            found += 1
            if found >= 3:
                break


if __name__ == "__main__":
    main()
