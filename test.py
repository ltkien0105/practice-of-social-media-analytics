import pandas as pd
import numpy as np
new = pd.read_csv('submission.csv')
old = pd.read_csv('M11415803_Le_Trung_Kien_sonnet5fold.csv')
print(f'Shapes:  new={new.shape}  old={old.shape}')
print(f'Columns: new={list(new.columns)}  old={list(old.columns)}')
print(f'IDs match: {(new["ID"].values == old["ID"].values).all()}')
diff = np.abs(new['Label'].values - old['Label'].values)
print(f'Label diff:  max={diff.max():.3e}  mean={diff.mean():.3e}')
print(f'Byte-identical: {diff.max() == 0}')