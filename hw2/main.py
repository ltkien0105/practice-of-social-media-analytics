"""
HW2 - Community membership prediction.

Build undirected graph from train.csv, partition with Louvain modularity
maximization at multiple resolutions, then for each test pair predict 1 if
both nodes belong to the same community (and same connected component) else 0.

A higher resolution -> more, smaller communities (fewer positive predictions).
A lower resolution  -> fewer, larger communities (more positive predictions).
One submission file per resolution; pick the best on the Kaggle leaderboard.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import community as community_louvain  # python-louvain
import networkx as nx

ROOT = Path(__file__).parent
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"

SEED = 42
RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def load_graph(path: Path) -> nx.Graph:
    g = nx.Graph()
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            u, v = int(row[0]), int(row[1])
            if u != v:
                g.add_edge(u, v)
    return g


def component_map(g: nx.Graph) -> dict[int, int]:
    """Return node -> connected-component id."""
    comp_of: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = cid
    return comp_of


def predict(test_path: Path, comp_of: dict[int, int],
            comm_of: dict[int, int]) -> list[tuple[int, int]]:
    preds: list[tuple[int, int]] = []
    with test_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idx, n1, n2 = int(row[0]), int(row[1]), int(row[2])
            # unknown node -> no training signal
            if n1 not in comp_of or n2 not in comp_of:
                preds.append((idx, 0))
                continue
            # different connected components -> definitely different community
            if comp_of[n1] != comp_of[n2]:
                preds.append((idx, 0))
                continue
            preds.append((idx, 1 if comm_of[n1] == comm_of[n2] else 0))
    return preds


def write_submission(preds: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        w.writerows(preds)


def run_resolution(g: nx.Graph, comp_of: dict[int, int],
                   resolution: float) -> tuple[int, float, int, Path]:
    t0 = time.time()
    comm_of = community_louvain.best_partition(
        g, resolution=resolution, random_state=SEED
    )
    n_comm = len(set(comm_of.values()))
    mod = community_louvain.modularity(comm_of, g)
    preds = predict(TEST, comp_of, comm_of)
    pos = sum(1 for _, c in preds if c == 1)
    out_path = ROOT / f"submission_res{resolution:g}.csv"
    write_submission(preds, out_path)
    print(f"  res={resolution:<5} communities={n_comm:<6} "
          f"modularity={mod:.4f}  positives={pos:<4} "
          f"-> {out_path.name} ({time.time() - t0:.1f}s)")
    return n_comm, mod, pos, out_path


def main() -> None:
    t0 = time.time()
    print("Loading graph...")
    g = load_graph(TRAIN)
    print(f"  nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
          f" ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("Computing connected components...")
    comp_of = component_map(g)
    n_comp = len(set(comp_of.values()))
    print(f"  components={n_comp} ({time.time() - t0:.1f}s)")

    print("\nResolution sweep:")
    for res in RESOLUTIONS:
        run_resolution(g, comp_of, res)

    print("\nDone. Submit each submission_res*.csv to Kaggle and keep the "
          "best-scoring resolution.")


if __name__ == "__main__":
    main()
