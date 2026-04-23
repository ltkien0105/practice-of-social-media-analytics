"""Graph-structural feature extraction for directed link prediction.

27 features per pair (u, v):
  - 10 neighbourhood overlap signals
  - 10 degree / structural ratios
  - 6 global rank scores  (PageRank + HITS hub/authority)
  - 1 community membership signal
"""

import math
import networkx as nx
import pandas as pd
from tqdm import tqdm

FEATURE_NAMES: list[str] = [
    # neighbourhood
    "out_in_common", "in_out_common", "common_succ", "common_pred",
    "jaccard_out_in", "adamic_adar", "resource_alloc_dir",
    "triadic_closure", "reverse_exists", "common_all",
    # degree
    "u_out_deg", "u_in_deg", "v_out_deg", "v_in_deg",
    "u_total_deg", "v_total_deg",
    "pref_attach", "pref_attach_total", "u_follow_ratio", "v_follow_ratio",
    # global rank
    "u_pagerank", "v_pagerank",
    "u_hub", "u_auth", "v_hub", "v_auth",
    # community
    "same_community",
]


def build_global_scores(G: nx.DiGraph) -> tuple[dict, dict, dict, dict, dict]:
    """Compute PageRank, HITS, and Louvain community membership."""
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    print("  Computing HITS...")
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)

    print("  Computing Louvain communities...")
    communities = nx.community.louvain_communities(G.to_undirected(), seed=42)
    node_community: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_community[n] = cid

    return pagerank, hubs, authorities, node_community


def compute_graph_features(
    pairs: list[tuple[int, int]],
    G: nx.DiGraph,
    pagerank: dict,
    hubs: dict,
    authorities: dict,
    node_community: dict,
    desc: str = "Graph features",
) -> pd.DataFrame:
    rows = []
    for u, v in tqdm(pairs, desc=desc, miniters=2000):
        out_u = set(G.successors(u)) if G.has_node(u) else set()
        in_u  = set(G.predecessors(u)) if G.has_node(u) else set()
        out_v = set(G.successors(v)) if G.has_node(v) else set()
        in_v  = set(G.predecessors(v)) if G.has_node(v) else set()

        out_in      = out_u & in_v
        in_out      = in_u & out_v
        common_succ = out_u & out_v
        common_pred = in_u & in_v

        union_out_in = out_u | in_v
        jaccard = len(out_in) / len(union_out_in) if union_out_in else 0.0

        aa = 0.0
        ra = 0.0
        for w in out_in:
            deg = G.out_degree(w) + G.in_degree(w)
            if deg > 1:
                aa += 1.0 / math.log(deg)
            out_w = G.out_degree(w)
            if out_w > 0:
                ra += 1.0 / out_w

        u_out = G.out_degree(u) if G.has_node(u) else 0
        u_in  = G.in_degree(u)  if G.has_node(u) else 0
        v_out = G.out_degree(v) if G.has_node(v) else 0
        v_in  = G.in_degree(v)  if G.has_node(v) else 0
        u_total = u_out + u_in
        v_total = v_out + v_in

        common_all    = len(out_in) + len(in_out) + len(common_succ) + len(common_pred)
        triadic       = len(out_in) / u_out if u_out > 0 else 0.0
        u_follow_rat  = u_out / (u_in + 1)
        v_follow_rat  = v_out / (v_in + 1)

        cu = node_community.get(u, -1)
        cv = node_community.get(v, -2)

        rows.append([
            len(out_in), len(in_out), len(common_succ), len(common_pred),
            jaccard, aa, ra, triadic, int(G.has_edge(v, u)), common_all,
            u_out, u_in, v_out, v_in, u_total, v_total,
            u_out * v_in, u_total * v_total, u_follow_rat, v_follow_rat,
            pagerank.get(u, 0.0), pagerank.get(v, 0.0),
            hubs.get(u, 0.0), authorities.get(u, 0.0),
            hubs.get(v, 0.0), authorities.get(v, 0.0),
            int(cu == cv and cu != -1),
        ])

    return pd.DataFrame(rows, columns=FEATURE_NAMES)
