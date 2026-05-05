"""Generate the 0.9/0.1 rank blend — predicted near-optimum."""
import pandas as pd
from scipy.stats import rankdata

m1 = pd.read_csv("M11415803_Le_Trung_Kien.csv")
m2 = pd.read_csv("M11415803_Le_Trung_Kien_0505_opus5fold.csv")
assert (m1["ID"] == m2["ID"]).all()

r1 = rankdata(m1["Label"].values) / len(m1)
r2 = rankdata(m2["Label"].values) / len(m2)

for w in [0.90, 0.85]:
    name = f"M11415803_Le_Trung_Kien_blend_rank_{int(w*100)}_{int((1-w)*100)}.csv"
    pd.DataFrame({"ID": m1["ID"], "Label": w * r1 + (1 - w) * r2}).to_csv(name, index=False)
    print(f"  {name}")
