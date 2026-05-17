"""
3-way ensemble: tuned + hidden + walktrap probabilities.

Walktrap classifier scores similarly to hidden but has slightly
different probability ordering. Blending all three may break the
0.948 plateau set by tuned+hidden alone.
"""

from __future__ import annotations

import numpy as np

from graph_utils import ROOT, load_test_pairs, write_submission

TARGET_COUNTS = [455, 458, 460, 462, 465, 468, 470]


def main() -> None:
    tuned = np.load(ROOT / "tuned_probs.npy")
    hidden = np.load(ROOT / "hidden_edge_probs.npy")
    walk = np.load(ROOT / "walktrap_probs.npy")
    test_pairs = load_test_pairs()

    # Rank correlations between all three
    rt = np.argsort(np.argsort(-tuned))
    rh = np.argsort(np.argsort(-hidden))
    rw = np.argsort(np.argsort(-walk))
    print(f"Corr tuned-hidden:    {np.corrcoef(rt, rh)[0, 1]:.4f}")
    print(f"Corr tuned-walktrap:  {np.corrcoef(rt, rw)[0, 1]:.4f}")
    print(f"Corr hidden-walktrap: {np.corrcoef(rh, rw)[0, 1]:.4f}")

    # Disagreement of walktrap with hidden in top-500
    h_top = set(np.argsort(-hidden)[:500].tolist())
    w_top = set(np.argsort(-walk)[:500].tolist())
    print(f"hidden vs walk top-500 disagreement: "
          f"{len(h_top ^ w_top) // 2} pairs")

    # Mean rank ensemble (Borda over all 3)
    rank_avg = (rt + rh + rw) / 3.0
    order = np.argsort(rank_avg)
    for n in TARGET_COUNTS:
        positive_idx = set(order[:n].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        out = ROOT / f"submission_3rankavg_top{n}.csv"
        write_submission(preds, out)
    print(f"  3-way rank-avg: {len(TARGET_COUNTS)} submissions")

    # Weighted blends (best from past: 0.3 tuned, 0.7 hidden; add walk)
    blends = [
        (0.2, 0.5, 0.3, "t20h50w30"),
        (0.2, 0.4, 0.4, "t20h40w40"),
        (0.3, 0.4, 0.3, "t30h40w30"),
        (0.25, 0.45, 0.30, "t25h45w30"),
        (0.0, 0.5, 0.5, "t00h50w50"),
        (0.1, 0.6, 0.3, "t10h60w30"),
        (0.15, 0.55, 0.30, "t15h55w30"),
        (0.30, 0.50, 0.20, "t30h50w20"),
    ]
    for wt, wh, ww, tag in blends:
        ens = wt * tuned + wh * hidden + ww * walk
        order = np.argsort(-ens)
        for n in TARGET_COUNTS:
            positive_idx = set(order[:n].tolist())
            preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                     for i in range(len(test_pairs))]
            out = ROOT / f"submission_3w_{tag}_top{n}.csv"
            write_submission(preds, out)
        print(f"  blend {tag}: {len(TARGET_COUNTS)} submissions")


if __name__ == "__main__":
    main()
