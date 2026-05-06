"""Diagnose model disagreement, then generate blend submissions."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

m1 = pd.read_csv("M11415803_Le_Trung_Kien_A.csv")
m2 = pd.read_csv("M11415803_Le_Trung_Kien_B.csv")
assert (m1["ID"] == m2["ID"]).all(), "ID order mismatch"

p1, p2 = m1["Label"].values, m2["Label"].values

print("=== Distributions ===")
print(f"main_1 (LB 0.85428): min={p1.min():.4g}  max={p1.max():.4g}  mean={p1.mean():.4f}  median={np.median(p1):.4f}")
print(f"main_2 (LB 0.78773): min={p2.min():.4g}  max={p2.max():.4g}  mean={p2.mean():.4f}  median={np.median(p2):.4f}")

print("\n=== Rank correlation (Spearman) ===")
rho, _ = spearmanr(p1, p2)
print(f"  rho = {rho:.4f}  (1.0 = identical ranking, 0 = uncorrelated)")

# Quantile of disagreement
q1 = np.quantile(p1, [0.5, 0.9, 0.99])
q2 = np.quantile(p2, [0.5, 0.9, 0.99])
print("\n=== Quantiles ===")
print(f"main_1 q50/q90/q99: {q1}")
print(f"main_2 q50/q90/q99: {q2}")

# Count strong disagreements (rank percentile difference > 0.5)
r1 = rankdata(p1) / len(p1)
r2 = rankdata(p2) / len(p2)
disagree = np.abs(r1 - r2) > 0.5
print(f"\nStrong disagreements (rank diff > 0.5): {disagree.sum()} / {len(p1)} ({disagree.mean()*100:.1f}%)")

# Generate blends
print("\n=== Generating blends ===")

def save(name, label):
    print(label)
    out = pd.DataFrame({"ID": m1["ID"], "Label": label})
    out.to_csv(name, index=False)
    print(f"  {name}  range=[{label.min():.4g}, {label.max():.4g}]")

# Rank-space blends (handles calibration mismatch)
save("M11415803_Le_Trung_Kien_blend_rank_80_20.csv", 0.8 * r1 + 0.2 * r2)
