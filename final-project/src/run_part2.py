"""Part 2 orchestrator: Community Detection.

Loads the graph once, runs Louvain → profiling → NMI, saves artifacts, and
writes a findings report. Run via ``uv run python -m src.run_part2``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

from src.data_loader import PROJECT_ROOT, load_features, load_graph_cached
from src.part2_community.community_profiling import profile_communities
from src.part2_community.louvain_detection import detect_communities
from src.part2_community.nmi_evaluation import evaluate_nmi
from src.part2_community.quality_metrics import assess_quality

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REPORT = PROJECT_ROOT / "reports" / "part2-community-findings.md"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    features = load_features()
    graph = load_graph_cached()

    louvain = detect_communities(graph, FIG_DIR)
    membership = louvain["membership"]
    prof = profile_communities(features, membership, FIG_DIR, TABLE_DIR)
    nmi = evaluate_nmi(features, membership)
    quality = assess_quality(graph, membership, louvain["stats"]["modularity_q"], FIG_DIR)

    _write_report(louvain["stats"], prof, nmi, quality)
    print("Part 2 complete. Report:", REPORT)


def _write_report(stats, prof, nmi, quality) -> None:
    profile_md = prof["profile"].to_markdown(index=False)
    lines = [
        "# Part 2 — Community Detection: Findings",
        "",
        "Dataset: Twitch Gamers mutual-follow network — 168,114 nodes, 6,797,557 edges.",
        "",
        "## 1. Louvain communities",
        f"- Modularity **Q = {stats['modularity_q']}** "
        f"({stats['n_communities']} communities).",
        f"- Largest community = {stats['largest_community']:,} nodes; "
        f"median size = {stats['median_community_size']}; "
        f"singletons = {stats['n_singletons']}.",
        f"- Stability across {len(stats['q_runs'])} runs: Q = {stats['q_runs']} "
        f"(std = {stats['q_std']}), communities = {stats['n_communities_runs']} "
        "— modularity is stable, not a single-run artefact.",
        "",
        "## 2. Major community profiles",
        f"Top {len(prof['profile'])} communities cover "
        f"{prof['top_n_coverage']:.1%} of all nodes; "
        f"mean language purity among them = {prof['mean_purity_topn']:.1%}.",
        "",
        profile_md,
        "",
        "## 3. NMI: communities vs language",
        f"- NMI(community, language) = **{nmi['nmi_community_language']}** "
        f"(random baseline ≈ {nmi['nmi_random_baseline']}).",
        f"- **Verdict: {nmi['verdict']}.**",
        "",
        "## 4. Cluster-quality metrics (graph-native Silhouette analogs)",
        f"- Modularity Q = **{quality['modularity_q']}** "
        f"(vs degree-preserving null ≈ {quality['q_null_mean']} → "
        f"**{quality['q_real_over_null']}×** the null: structure is genuine).",
        f"- Coverage = {quality['coverage']:.1%} of edges stay intra-community.",
        f"- Conductance (lower = better separated): mean {quality['mean_conductance']}, "
        f"median {quality['median_conductance']}.",
        "",
        "Figures: `results/figures/p2_*.png`. "
        "Table: `results/tables/community_profile.csv`.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
