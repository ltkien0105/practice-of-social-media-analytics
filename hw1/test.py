from scipy.stats import rankdata, spearmanr
import numpy as np

def test_rankdata():
    x1 = [0.1, 0.5, 0.3, 0.2]
    x2 = [0.2, 0.7, 0.4, 0.3]
    print(rankdata(x1))
    print(rankdata(x2))
    r1 = rankdata(x1) / len(x1)
    r2 = rankdata(x2) / len(x2)
    print("Ranks for x1:", r1)
    print("Ranks for x2:", r2)
    print(0.8 * r1)
    print(0.2 * r2)
    print(0.2 * r1 + 0.8 * r2)
    # disagree = np.abs(r1 - r2) > 0.5
    # print(disagree)
    # print(f"Strong disagreements (rank diff > 0.5): {disagree.sum()} / {len(x1)} ({disagree.mean()*100:.1f}%)")
    # # assert (r1 == [1, 4, 3, 2]).all(), f"Expected ranks [1,4,3,2], got {r1}"
    # # assert (r2 == [2, 3, 1, 4]).all(), f"Expected ranks [2,3,1,4], got {r2}"

test_rankdata()