"""Shared utilities: graph loading, component map, prediction, writing."""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).parent
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"


def load_edges(path: Path = TRAIN) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            if u != v:
                edges.append((u, v))
    return edges


def load_graph(path: Path = TRAIN) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(load_edges(path))
    return g


def component_map(g: nx.Graph) -> dict[int, int]:
    comp_of: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = cid
    return comp_of


def load_test_pairs(path: Path = TEST) -> list[tuple[int, int, int]]:
    """Return list of (id, n1, n2)."""
    pairs: list[tuple[int, int, int]] = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pairs.append((int(row[0]), int(row[1]), int(row[2])))
    return pairs


def predict_same_community(test_pairs: list[tuple[int, int, int]],
                           comp_of: dict[int, int],
                           comm_of: dict[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, n1, n2 in test_pairs:
        if n1 not in comp_of or n2 not in comp_of:
            out.append((idx, 0))
            continue
        if comp_of[n1] != comp_of[n2]:
            out.append((idx, 0))
            continue
        out.append((idx, 1 if comm_of[n1] == comm_of[n2] else 0))
    return out


def write_submission(preds: list[tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        w.writerows(preds)
