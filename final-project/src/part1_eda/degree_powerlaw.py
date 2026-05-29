"""Degree distribution and power-law fit.

Fits a power law to the degree sequence, reports the exponent alpha plus a
goodness-of-fit check, and compares against a lognormal alternative so we do not
over-claim "scale-free" on a distribution that is merely heavy-tailed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import powerlaw


def analyze_degree_powerlaw(degrees: np.ndarray, fig_dir: Path) -> dict:
    degrees = np.asarray(degrees)
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    ks = fit.power_law.D  # Kolmogorov-Smirnov distance
    # R > 0 favors power law, R < 0 favors lognormal; p is significance of the sign.
    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)

    # Scale-free verdict: tail exponent in the canonical (2, 3) band AND power law
    # not significantly rejected in favor of lognormal.
    scale_free = (2.0 < alpha < 3.0) and not (R < 0 and p < 0.05)

    _plot_ccdf(degrees, fit, fig_dir)

    return {
        "alpha": round(float(alpha), 4),
        "xmin": int(xmin),
        "ks_distance": round(float(ks), 4),
        "loglik_ratio_vs_lognormal": round(float(R), 4),
        "loglik_p": round(float(p), 4),
        "max_degree": int(degrees.max()),
        "mean_degree": round(float(degrees.mean()), 2),
        "scale_free": bool(scale_free),
    }


def _plot_ccdf(degrees: np.ndarray, fit: "powerlaw.Fit", fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fit.plot_ccdf(ax=ax, color="steelblue", linewidth=1.5, label="empirical CCDF")
    fit.power_law.plot_ccdf(
        ax=ax, color="crimson", linestyle="--", label=f"power-law fit (α={fit.power_law.alpha:.2f})"
    )
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title("Degree distribution (log-log CCDF) with power-law fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_degree_powerlaw_ccdf.png", dpi=150)
    plt.close(fig)
