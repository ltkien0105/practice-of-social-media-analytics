"""NMI between detected communities and language labels.

Normalized Mutual Information quantifies how much the Louvain partition agrees
with the language partition. High NMI → the network is organised by language;
low NMI → other factors (follow-cliques, geography, topic) dominate. A random
baseline contextualises the value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def evaluate_nmi(df: pd.DataFrame, membership: np.ndarray) -> dict:
    language_codes = pd.Categorical(df["language"]).codes
    nmi = normalized_mutual_info_score(membership, language_codes)

    # Baseline: NMI against a deterministic shuffle of the community labels.
    # A genuine signal should sit far above this near-zero baseline.
    shuffled = np.sort(membership)  # destroys node<->label correspondence
    baseline = normalized_mutual_info_score(shuffled, language_codes)

    if nmi >= 0.5:
        verdict = "strongly organised by language"
    elif nmi >= 0.2:
        verdict = "partially organised by language"
    else:
        verdict = "NOT primarily organised by language (other factors dominate)"

    return {
        "nmi_community_language": round(float(nmi), 4),
        "nmi_random_baseline": round(float(baseline), 4),
        "verdict": verdict,
    }
