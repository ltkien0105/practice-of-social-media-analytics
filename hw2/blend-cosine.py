"""
Blend baseline probs (tuned + hidden) with raw Node2Vec cosine similarity.
The hadamard classifier hurt (0.668); cosine is good (0.946). Use cosine.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
TEST_CSV = ROOT / "test.csv"

tuned = np.load(ROOT / "tuned_probs.npy")
hidden = np.load(ROOT / "hidden_probs.npy")
emb = np.load(ROOT / "n2v_embeddings.npy")
with (ROOT / "n2v_node_index.pkl").open("rb") as f:
    idx = pickle.load(f)

test_pairs: list[tuple[int, int, int]] = []
with TEST_CSV.open() as f:
    r = csv.reader(f); next(r)
    for row in r:
        test_pairs.append((int(row[0]), int(row[1]), int(row[2])))

norms = np.linalg.norm(emb, axis=1)
sims = np.zeros(len(test_pairs))
for i, (_, u, v) in enumerate(test_pairs):
    iu, iv = idx.get(u), idx.get(v)
    if iu is None or iv is None:
        continue
    nu, nv = norms[iu], norms[iv]
    if nu == 0 or nv == 0:
        continue
    sims[i] = float(np.dot(emb[iu], emb[iv]) / (nu * nv))

base = 0.3 * tuned + 0.7 * hidden


def topn_write(scores, name, n=460):
    order = np.argsort(-scores)
    pos = set(order[:n].tolist())
    out_path = ROOT / name
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for i, (tid, _, _) in enumerate(test_pairs):
            w.writerow([tid, 1 if i in pos else 0])
    base_set = set(np.argsort(-base)[:n].tolist())
    print(f"  {name}: positives={len(pos)} overlap_with_baseline="
          f"{len(pos & base_set)}/{n}")


print("Blending baseline + cosine at various weights (top-460):")
for w_cos in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    ens = (1 - w_cos) * base + w_cos * sims
    topn_write(ens, f"submission_blend_cos{int(w_cos * 100):02d}_top460.csv")

print("\nRank-average baseline + cosine (rank-based, threshold-free):")
rank_base = np.argsort(np.argsort(base)).astype(float) / (len(base) - 1)
rank_cos = np.argsort(np.argsort(sims)).astype(float) / (len(sims) - 1)
for w_cos in (0.10, 0.20, 0.30, 0.40, 0.50):
    ens = (1 - w_cos) * rank_base + w_cos * rank_cos
    topn_write(ens, f"submission_rankblend_cos{int(w_cos * 100):02d}_top460.csv")
