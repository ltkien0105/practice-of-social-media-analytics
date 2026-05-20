"""
Node2Vec experiment — tries to push past the 0.948 plateau.

Pipeline:
1. Build graph from train.csv.
2. Generate random walks (DeepWalk, p=q=1) — uniform neighbor sampling.
3. Train Word2Vec SGNS to get node embeddings (cached to .npy).
4. Two predictors:
   (a) cos_sim — pure cosine similarity, top-N submission.
   (b) hadamard + GBM classifier — supervised on edge/cross-leiden labels.
5. Also blend predictor (b) with main.py's tuned+hidden probs at multiple
   weights and produce candidate submissions for Kaggle testing.

Run:
    uv run python node2vec_experiment.py
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from pathlib import Path

import community as community_louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

SEED = 42
EMB_DIM = 128
WALK_LEN = 80
N_WALKS = 10
WINDOW = 10
EPOCHS = 5
WORKERS = 8
MIN_COUNT = 0

# Node2Vec biased-walk parameters.
# p=1, q=1   -> DeepWalk (uniform). Run already cached at 0.946 cosine.
# p=1, q=0.5 -> DFS-bias (homophily / community-aware). This run.
P = 1.0
Q = 0.5

SUFFIX = f"_p{P}_q{Q}".replace(".", "")
EMB_PATH = ROOT / f"n2v_embeddings{SUFFIX}.npy"
NODE_INDEX_PATH = ROOT / f"n2v_node_index{SUFFIX}.pkl"
TAG = f"n2vq{int(Q * 100):02d}"  # used in submission filenames


def load_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with TRAIN_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            u, v = int(row[0]), int(row[1])
            if u != v:
                edges.append((u, v))
    return edges


def load_test_pairs() -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    with TEST_CSV.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            pairs.append((int(row[0]), int(row[1]), int(row[2])))
    return pairs


def write_submission(preds: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        w.writerows(preds)


def deepwalk_walks(adj: dict[int, list[int]], nodes: list[int],
                   n_walks: int, walk_len: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    walks: list[list[str]] = []
    for it in range(n_walks):
        t0 = time.time()
        order = nodes[:]
        rng.shuffle(order)
        for start in order:
            walk = [start]
            cur = start
            for _ in range(walk_len - 1):
                nbrs = adj.get(cur)
                if not nbrs:
                    break
                cur = nbrs[rng.randint(0, len(nbrs) - 1)]
                walk.append(cur)
            walks.append([str(n) for n in walk])
        print(f"  walk batch {it + 1}/{n_walks} ({time.time() - t0:.1f}s)")
    return walks


def biased_walks(adj: dict[int, list[int]],
                 nbr_set: dict[int, set[int]],
                 nodes: list[int], n_walks: int, walk_len: int,
                 p: float, q: float, seed: int) -> list[list[str]]:
    """Node2Vec biased walks. weight = 1/p if x==prev, 1 if x in N(prev), 1/q else."""
    rng = random.Random(seed)
    w_return = 1.0 / p
    w_dfs = 1.0 / q
    walks: list[list[str]] = []
    for it in range(n_walks):
        t0 = time.time()
        order = nodes[:]
        rng.shuffle(order)
        for start in order:
            walk = [start]
            prev = None
            cur = start
            for _ in range(walk_len - 1):
                nbrs = adj.get(cur)
                if not nbrs:
                    break
                if prev is None:
                    nxt = nbrs[rng.randint(0, len(nbrs) - 1)]
                else:
                    pn = nbr_set[prev]
                    weights = [
                        w_return if x == prev
                        else 1.0 if x in pn
                        else w_dfs
                        for x in nbrs
                    ]
                    nxt = rng.choices(nbrs, weights=weights, k=1)[0]
                walk.append(nxt)
                prev, cur = cur, nxt
            walks.append([str(n) for n in walk])
        print(f"  walk batch {it + 1}/{n_walks} ({time.time() - t0:.1f}s)")
    return walks


def train_embeddings(edges: list[tuple[int, int]]) -> tuple[np.ndarray, dict[int, int]]:
    """Train Node2Vec and return (embedding matrix, node->row index)."""
    if EMB_PATH.exists() and NODE_INDEX_PATH.exists():
        print(f"Loading cached embeddings from {EMB_PATH.name}...")
        emb = np.load(EMB_PATH)
        with NODE_INDEX_PATH.open("rb") as f:
            idx = pickle.load(f)
        return emb, idx

    print("Building adjacency dict...")
    adj: dict[int, list[int]] = {}
    nodes_set: set[int] = set()
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        nodes_set.add(u); nodes_set.add(v)
    nodes = sorted(nodes_set)
    nbr_set: dict[int, set[int]] = {n: set(a) for n, a in adj.items()}
    print(f"  nodes={len(nodes)} edges={len(edges)}")

    biased = (P != 1.0 or Q != 1.0)
    print(f"Generating {N_WALKS} x {len(nodes)} walks of length {WALK_LEN} "
          f"({'biased p=' + str(P) + ' q=' + str(Q) if biased else 'uniform'})...")
    t0 = time.time()
    if biased:
        walks = biased_walks(adj, nbr_set, nodes, N_WALKS, WALK_LEN, P, Q, SEED)
    else:
        walks = deepwalk_walks(adj, nodes, N_WALKS, WALK_LEN, SEED)
    print(f"  {len(walks)} walks generated ({time.time() - t0:.1f}s)")

    print(f"Training Word2Vec (dim={EMB_DIM}, window={WINDOW}, "
          f"epochs={EPOCHS}, workers={WORKERS})...")
    t0 = time.time()
    model = Word2Vec(
        sentences=walks,
        vector_size=EMB_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED,
        negative=10,
    )
    print(f"  trained ({time.time() - t0:.1f}s)")

    idx = {n: i for i, n in enumerate(nodes)}
    emb = np.zeros((len(nodes), EMB_DIM), dtype=np.float32)
    for n in nodes:
        key = str(n)
        if key in model.wv:
            emb[idx[n]] = model.wv[key]
    np.save(EMB_PATH, emb)
    with NODE_INDEX_PATH.open("wb") as f:
        pickle.dump(idx, f)
    print(f"  saved -> {EMB_PATH.name}")
    return emb, idx


def cosine_sims(emb: np.ndarray, idx: dict[int, int],
                pairs: list[tuple[int, int, int]]) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1)
    sims = np.zeros(len(pairs))
    for i, (_, u, v) in enumerate(pairs):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            sims[i] = 0.0
            continue
        nu, nv = norms[iu], norms[iv]
        if nu == 0 or nv == 0:
            sims[i] = 0.0
            continue
        sims[i] = float(np.dot(emb[iu], emb[iv]) / (nu * nv))
    return sims


def build_leiden(edges: list[tuple[int, int]]) -> dict[int, int]:
    nodes = sorted({n for e in edges for n in e})
    idx_of = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes),
                    edges=[(idx_of[u], idx_of[v]) for u, v in edges],
                    directed=False)
    part = la.find_partition(ig_g, la.ModularityVertexPartition, seed=SEED)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def component_map(g: nx.Graph) -> dict[int, int]:
    cm: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            cm[n] = cid
    return cm


def cross_leiden_negatives(edges: list[tuple[int, int]],
                           leiden_of: dict[int, int],
                           comp_of: dict[int, int],
                           n_neg: int, rng: random.Random) -> list[tuple[int, int]]:
    by_comp: dict[int, list[int]] = {}
    for n, c in comp_of.items():
        by_comp.setdefault(c, []).append(n)
    big = [c for c, ns in by_comp.items() if len(ns) >= 50]
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    neg: list[tuple[int, int]] = []
    attempts = 0
    while len(neg) < n_neg and attempts < n_neg * 30:
        attempts += 1
        c = rng.choice(big)
        ns = by_comp[c]
        u, v = rng.sample(ns, 2)
        if (min(u, v), max(u, v)) in edge_set:
            continue
        if leiden_of[u] == leiden_of[v]:
            continue
        neg.append((u, v))
    return neg


def hadamard(emb: np.ndarray, idx: dict[int, int],
             pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((len(pairs), emb.shape[1]), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        out[i] = emb[iu] * emb[iv]
    return out


def train_hadamard_classifier(emb: np.ndarray, idx: dict[int, int],
                              edges: list[tuple[int, int]],
                              leiden_of: dict[int, int],
                              comp_of: dict[int, int],
                              test_pairs: list[tuple[int, int, int]],
                              n_ensemble: int = 6) -> np.ndarray:
    test_pp = [(n1, n2) for _, n1, n2 in test_pairs]
    Xtest = hadamard(emb, idx, test_pp)
    probs = np.zeros(len(test_pairs))
    for k in range(n_ensemble):
        t0 = time.time()
        seed = SEED + k * 17
        rng = random.Random(seed)
        pos = rng.sample(edges, 5000)
        neg = cross_leiden_negatives(edges, leiden_of, comp_of, 5000, rng)
        X = hadamard(emb, idx, pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        random_state=seed)
        gb.fit(X, y)
        probs += gb.predict_proba(Xtest)[:, 1]
        print(f"  n2v-clf member {k + 1}/{n_ensemble} "
              f"({time.time() - t0:.1f}s)")
    return probs / n_ensemble


def topn_submission(scores: np.ndarray,
                    test_pairs: list[tuple[int, int, int]],
                    top_n: int, name: str) -> None:
    order = np.argsort(-scores)
    positive_idx = set(order[:top_n].tolist())
    preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
             for i in range(len(test_pairs))]
    pos_count = sum(1 for _, c in preds if c == 1)
    path = ROOT / name
    write_submission(preds, path)
    print(f"  positives={pos_count} -> {path.name}")


def thresh_submission(scores: np.ndarray,
                      test_pairs: list[tuple[int, int, int]],
                      thresh: float, name: str) -> None:
    preds = [(test_pairs[i][0], 1 if scores[i] >= thresh else 0)
             for i in range(len(test_pairs))]
    pos_count = sum(1 for _, c in preds if c == 1)
    path = ROOT / name
    write_submission(preds, path)
    print(f"  positives={pos_count} -> {path.name}")


def main() -> None:
    t_total = time.time()
    edges = load_edges()
    test_pairs = load_test_pairs()
    print(f"edges={len(edges)} test_pairs={len(test_pairs)}")

    emb, idx = train_embeddings(edges)
    print(f"embeddings: {emb.shape}")

    print("\nGraph metadata for label generation...")
    g = nx.Graph(); g.add_edges_from(edges)
    comp_of = component_map(g)
    leiden_of = build_leiden(edges)
    print(f"  leiden communities={len(set(leiden_of.values()))}")

    print("\nCosine similarity scores (pure unsupervised)...")
    sims = cosine_sims(emb, idx, test_pairs)
    print(f"  sim stats: min={sims.min():.3f} max={sims.max():.3f} "
          f"mean={sims.mean():.3f}")
    for top in (440, 460, 480, 500):
        topn_submission(sims, test_pairs, top,
                        f"submission_{TAG}_cos_top{top}.csv")
    for th in (0.5, 0.6, 0.7):
        thresh_submission(sims, test_pairs, th,
                          f"submission_{TAG}_cos_t{th}.csv")

    print("\nHadamard + GBM classifier ensemble...")
    n2v_probs = train_hadamard_classifier(emb, idx, edges, leiden_of,
                                          comp_of, test_pairs, n_ensemble=6)
    np.save(ROOT / f"{TAG}_probs.npy", n2v_probs)
    print(f"  probs stats: min={n2v_probs.min():.3f} "
          f"max={n2v_probs.max():.3f} mean={n2v_probs.mean():.3f}")
    for top in (440, 460, 480, 500):
        topn_submission(n2v_probs, test_pairs, top,
                        f"submission_{TAG}_clf_top{top}.csv")
    for th in (0.5, 0.6):
        thresh_submission(n2v_probs, test_pairs, th,
                          f"submission_{TAG}_clf_t{th}.csv")

    tuned_path = ROOT / "tuned_probs.npy"
    hidden_path = ROOT / "hidden_probs.npy"
    if tuned_path.exists() and hidden_path.exists():
        print("\nBlending baseline + biased-n2v cosine at multiple weights...")
        tuned = np.load(tuned_path)
        hidden = np.load(hidden_path)
        base = 0.3 * tuned + 0.7 * hidden
        rank_base = np.argsort(np.argsort(base)).astype(float) / (len(base) - 1)
        rank_cos = np.argsort(np.argsort(sims)).astype(float) / (len(sims) - 1)
        for w in (0.10, 0.20, 0.30, 0.50):
            ens = (1 - w) * base + w * sims
            topn_submission(
                ens, test_pairs, 460,
                f"submission_blend_{TAG}cos{int(w * 100):02d}_top460.csv",
            )
        for w in (0.30, 0.50):
            ens = (1 - w) * rank_base + w * rank_cos
            topn_submission(
                ens, test_pairs, 460,
                f"submission_rankblend_{TAG}cos{int(w * 100):02d}_top460.csv",
            )
    else:
        print("\nSkip blend; run main.py to cache baseline probs first.")

    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
