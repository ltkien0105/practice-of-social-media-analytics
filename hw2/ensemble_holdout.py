"""
Ensemble including the saturated holdout probabilities.

Holdout's probs are saturated (median 0.996), so its ordering within
high-prob pairs may still differ from hidden/tuned. Useful for
tie-breaking at the top-N cutoff.
"""

from __future__ import annotations

import numpy as np

from graph_utils import ROOT, load_test_pairs, write_submission

TARGET_COUNTS = [455, 458, 460, 462, 465]


def main() -> None:
    tuned = np.load(ROOT / "tuned_probs.npy")
    hidden = np.load(ROOT / "hidden_edge_probs.npy")
    holdout = np.load(ROOT / "holdout_probs.npy")
    test_pairs = load_test_pairs()

    rt = np.argsort(np.argsort(-tuned))
    rh = np.argsort(np.argsort(-hidden))
    rho = np.argsort(np.argsort(-holdout))
    print(f"Corr holdout vs hidden: {np.corrcoef(rho, rh)[0, 1]:.4f}")
    print(f"Corr holdout vs tuned:  {np.corrcoef(rho, rt)[0, 1]:.4f}")

    h_top = set(np.argsort(-hidden)[:500].tolist())
    ho_top = set(np.argsort(-holdout)[:500].tolist())
    print(f"hidden vs holdout top-500 disagreement: "
          f"{len(h_top ^ ho_top) // 2}")

    # 4-way rank average
    rank_avg = (rt + rh + rho + rho) / 4.0  # double-weight holdout
    order = np.argsort(rank_avg)
    for n in TARGET_COUNTS:
        positive_idx = set(order[:n].tolist())
        preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                 for i in range(len(test_pairs))]
        out = ROOT / f"submission_4rank_top{n}.csv"
        write_submission(preds, out)

    # Weighted blends
    blends = [
        (0.3, 0.5, 0.2, "ho20"),
        (0.2, 0.4, 0.4, "ho40"),
        (0.2, 0.5, 0.3, "ho30"),
        (0.15, 0.55, 0.30, "ho30b"),
        (0.0, 0.5, 0.5, "h0_ho50"),
    ]
    for wt, wh, wo, tag in blends:
        ens = wt * tuned + wh * hidden + wo * holdout
        order = np.argsort(-ens)
        for n in TARGET_COUNTS:
            positive_idx = set(order[:n].tolist())
            preds = [(test_pairs[i][0], 1 if i in positive_idx else 0)
                     for i in range(len(test_pairs))]
            out = ROOT / f"submission_holdblend_{tag}_top{n}.csv"
            write_submission(preds, out)
        print(f"  blend {tag}: written")


if __name__ == "__main__":
    main()
