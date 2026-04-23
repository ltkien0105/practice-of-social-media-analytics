"""Negative edge sampling strategies for link prediction training."""

import random
import networkx as nx


def sample_random_negatives(G: nx.DiGraph, n: int, seed: int = 42) -> list[tuple]:
    """Uniform random non-edges."""
    nodes = list(G.nodes())
    edge_set = set(G.edges())
    rng = random.Random(seed)
    negs: list[tuple] = []
    while len(negs) < n:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and (u, v) not in edge_set:
            negs.append((u, v))
    return negs


def sample_hard_negatives(G: nx.DiGraph, n: int, seed: int = 42) -> list[tuple]:
    """2-hop walk non-edges — harder to distinguish from true edges.

    Samples (u, v) where v is reachable in exactly 2 hops from u but no
    direct edge (u→v) exists.  Falls back to random sampling to fill quota.
    """
    edge_set = set(G.edges())
    rng = random.Random(seed)
    negs: set[tuple] = set()
    nodes = list(G.nodes())

    attempts = 0
    while len(negs) < n and attempts < n * 5:
        attempts += 1
        u = rng.choice(nodes)
        out_u = list(G.successors(u))
        if not out_u:
            continue
        w = rng.choice(out_u)
        out_w = list(G.successors(w))
        if not out_w:
            continue
        v = rng.choice(out_w)
        if v != u and (u, v) not in edge_set:
            negs.add((u, v))

    result = list(negs)
    # fill remainder with random if 2-hop pool exhausted
    while len(result) < n:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and (u, v) not in edge_set and (u, v) not in negs:
            result.append((u, v))
    return result[:n]


def sample_mixed_negatives(
    G: nx.DiGraph, n: int, hard_ratio: float = 0.3, seed: int = 42,
) -> list[tuple]:
    """Mix of hard (2-hop) and random negatives for a robust training set."""
    n_hard = int(n * hard_ratio)
    n_rand = n - n_hard
    hard = sample_hard_negatives(G, n_hard, seed=seed)
    rand = sample_random_negatives(G, n_rand, seed=seed + 1)
    return hard + rand
