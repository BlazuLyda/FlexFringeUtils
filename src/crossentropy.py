import numpy as np


qs = np.array([0.01, 0.005, 0.03, 0.0007, 0.0004, 0.01, 0.003])
ps = np.array([0.015, 0.005, 0.023, 0.001, 0.0003, 0.008, 0.0025])

qs_norm = qs / qs.sum()
ps_norm = ps / ps.sum()


ce_1 = -1/qs.sum() * np.sum(qs * np.log(ps)) + np.log(ps.sum())
ce_2 = -np.sum(qs_norm * np.log(ps_norm))

print(ce_1)
print(ce_2)
assert ce_1 == ce_2
