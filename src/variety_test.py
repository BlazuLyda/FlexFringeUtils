import variety as var
import numpy as np

# === Test functions ===
def test_remove_outliers_from_scores_flat():
    scores = [0.0, 1, 2, 3,
              4, 5, 6, 7,
              8, 9, 10, 11,
              12, 13, 14, 15]  # 4x4 matrix
    n = 4
    outliers = [1]
    expected = [
        0, 2, 3,
        8, 10, 11,
        12, 14, 15
    ]  # Corresponds to removing row/col 1
    result = var.remove_outliers_from_scores_flat(scores, n, outliers)
    assert result == expected


def test_reindex_with_outliers():
    original = [0, 1, 3]
    outliers = [1, 2]

    expected = [0, 3, 5]
    result = var.reindex_with_outliers(original, outliers, 10)
    assert result == expected


def test_grow_ens_size():

    H_arr = [
        0.0, 1.0, 3.0, 2.8, 1.0,
        1.0, 0.0, 3.0, 2.8, 1.0,
        1.0, 2.0, 0.0, 4.0, 1.0,
        1.0, 2.5, 3.0, 0.0, 1.0,
        1.0, 2.5, 3.0, 1.0, 0.0
    ]
    H = np.array(H_arr)
    H.resize((5, 5))

    sel_prim = var.normalize_pruned_size_with_imv(H, m = 5, n = 4, sel = [0, 1])
    print(sel_prim)
    assert sel_prim == [0, 1, 2, 3]


def test_shrink_ens_size():

    H_arr = [
        0.0, 1.0, 3.0, 2.8, 1.0,
        1.0, 0.0, 3.0, 2.8, 1.0,
        1.0, 2.0, 0.0, 4.0, 1.0,
        1.0, 2.5, 3.0, 0.0, 1.0,
        1.0, 2.5, 3.0, 1.0, 0.0
    ]
    H = np.array(H_arr)
    H.resize((5, 5))

    sel_prim = var.normalize_pruned_size_with_imv(H, m = 5, n = 2, sel = [0, 1, 2, 3])
    assert sel_prim == [2, 3]