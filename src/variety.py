import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import AffinityPropagation
import kmedoids
from typing import Any

def scores_flat_to_normalized_cross_entropy(scores_flat: list[float], n: int) -> np.ndarray:

    # Flat array into matrix
    M = np.array(scores_flat, dtype=float).reshape((n, n))
    # Now normalize by subtracting the values from diagonal from each row
    H =  M - np.diag(M)[:, np.newaxis]
    return H


def compute_distance(C: np.ndarray, i: int, j: int) -> float:

    # Similarity index is:       Sim(i, j) = (C_i[i] + C_j[j]) / (C_i[j] + C_j[i])
    # Dissimilarity index is:    Dis(i, j) = 1 / Sim(i, j) - 1

    return (C[i, j] + C[j, i]) / (C[i, i] + C[j, j]) - 1


def compute_similarity(C: np.ndarray, i: int, j: int) -> float:

    # Similarity index is:       Sim(i, j) = (C_i[i] + C_j[j]) / (C_i[j] + C_j[i])

    return (C[i, i] + C[j, j]) / (C[i, j] + C[j, i])


def scores_flat_to_distance_matrix_gao(scores_flat: list[float], n: int) -> np.ndarray:

    # Get the sample cross entropy matrix
    C = np.array(scores_flat, dtype=float).reshape((n, n))

    # Now compute the Distance matrix according to Gao et al. 2021
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist = compute_distance(C, i, j)
            D[i, j] = dist
            D[j, i] = dist
    return D


def scores_flat_to_similarity_matrix_gao(scores_flat: list[float], n: int) -> np.ndarray:

    # Get the sample cross entropy matrix
    C = np.array(scores_flat, dtype=float).reshape((n, n))

    # Now compute the Similarity matrix according to Gao et al. 2021
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist = compute_similarity(C, i, j)
            D[i, j] = dist
            D[j, i] = dist
    return D


def scores_flat_to_distance_matrix_simple(scores_flat: list[float], n: int) -> np.ndarray:

    # Get the sample cross entropy matrix
    C = np.array(scores_flat, dtype=float).reshape((n, n))

    # Normalize by subtracting teh values from the diagonal from each row
    H =  C - np.diag(C)[:, np.newaxis]

    # Make the matrix symmetric
    return H + H.T


def compute_imv_weighted(H: np.ndarray, w: np.ndarray, n: int):
    """
    Computes the Inter Model Variety metric given a normalized cross entropy matrix and a weight vector.
    """
    assert H.shape == (n, n), f"H must be an n by n matrix: {H}"
    assert w.shape == (n,), f"Weights must be a vector of length n: {w}"
    assert np.all(w >= 0), f"Weights must be non-negative: {w}"
    assert np.isclose(w.sum(), 1.0), f"Weights must sum up to 1.0: {w}"

    # Method A) Compute weighted sum via outer product
    w_outer = np.outer(w, w)
    return np.sum(w_outer * H)

    # Method B) Compute weighted sum by applying the weights to the rows
    # H_weighted = H * w[:, np.newaxis]  # Broadcast w along columns
    # return np.sum(H_weighted)


def compute_imv(H: np.ndarray, n: int):
    """
    Computes the Inter Model Variety metric given a normalized cross entropy matrix.
    """
    assert H.shape == (n, n), f"H must be an n by n matrix: {H}"

    # Run imv with equal weights
    weights = np.ones(n) / n
    return compute_imv_weighted(H, weights, n)


def optimize_weights_for_imv(H: np.ndarray, n: int):

    assert H.shape == (n, n), "H must be an n by n matrix"

    # Objective: negative IMV (because we minimize)
    def objective_outer_imv(w):
        w_outer = np.outer(w, w)
        return -np.sum(w_outer * H)

    # Constraints: weights sum to 1
    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })

    # Bounds: weights are non-negative
    bounds = [(0, 1) for _ in range(n)]

    # Initial guess: uniform weights
    w0 = np.full(n, 1.0 / n)

    result = minimize(objective_outer_imv, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x  # optimal weights


def prune_ensemble_max_imv(H: np.ndarray, m: int, n: int) -> list[int]:
    """
    Progressively prunes a large ensemble from size m to size n by removing models with 
    the lowest (row + column) sum in the cross-entropy matrix H.
    Args:
        H: The m x m cross-entropy matrix.
        m: Original ensemble size.
        n: Target ensemble size (n < m).
    Returns:
        Indices of the selected models for the final ensemble of size n and the final pruned 
        cross entropy matrix of size n x n.
    """
    assert H.shape == (m, m), "H must be an m x m matrix"
    assert n <= m, "Target size n must be smaller than m"

    # Start with all indices
    remaining = list(range(m))
    H_current = H.copy()

    while len(remaining) > n:
        # Compute row + column sums for current matrix
        scores = H_current.sum(axis=0) + H_current.sum(axis=1) - H_current.diagonal()
        # Find index with lowest total contribution
        remove_idx = int(np.argmin(scores))
        # Remove the model
        remaining.pop(remove_idx)
        # Remove corresponding row and column from matrix
        H_current = np.delete(H_current, remove_idx, axis=0)
        H_current = np.delete(H_current, remove_idx, axis=1)

    return remaining


def compute_imv_scores(H: np.ndarray, n: int) -> list[float]:
    """
    Computes the IMV score of each model in the ensemble as the sum of the model
    row and column in the sample cross entropy matrix.
    """
    assert H.shape == (n, n), "H must be an n x n matrix"

    scores = H.sum(axis=0) + H.sum(axis=1) - H.diagonal()
    return scores


def prune_ensemble_kmedoids(D: np.ndarray, m: int, n: int) -> list[int]:
    """
    Select n representants from a group of models of size m using
    K-Medoids clustering. Uses distance matrix D for computations.
    """
    assert D.shape == (m, m), "H must be an m x m matrix"
    assert n <= m, "Target size n must be smaller than m"

    km = kmedoids.KMedoids(n_clusters=n, method='fasterpam', metric='precomputed')
    km.fit(D)
    return km.medoid_indices_


def prune_ensemble_affinity(S: np.ndarray, m: int, n: int) -> list[int]:
    """
    Select representative models from an ensemble of size m using
    Affinity Propagation clustering. Uses similarity matrix S.
    
    The target number of representants `n` is treated as a suggestion only.
    
    Parameters:
    - S: similarity matrix of shape (m, m), where higher values indicate more similarity.
    - m: number of models in the ensemble (S should be m x m).
    - n: suggested number of representative models (not enforced).

    Returns:
    - List of indices corresponding to selected exemplar models.
    """
    assert S.shape == (m, m), "S must be an m x m similarity matrix"
    assert n <= m, "Target size n must be smaller than m"

    af = AffinityPropagation(affinity='precomputed', random_state=42)
    af.fit(S)
    exemplars = af.cluster_centers_indices_
    return exemplars.tolist()



def cluster_ensemble_kmedoids(D: np.ndarray, m: int, n: int) -> tuple[list[list[int]], list[int]]:
    km = kmedoids.KMedoids(n_clusters=n, method='fasterpam', metric='precomputed')
    km.fit(D)
    return (km.labels_, km.medoid_indices_)


def cluster_ensemble_affinity(S: np.ndarray, m: int, n: int) -> Any:
    af = AffinityPropagation(affinity='precomputed', random_state=42)
    af.fit(S)
    return (af.labels_, af.cluster_centers_indices_)


def normalize_pruned_size_with_imv(H: np.ndarray, m: int, n: int, sel: list[int]) -> list[int]:
    l = len(sel)

    if l == n:
        return sel
    if l > n:
        # Reduce the size of the ensemble
        H_l = H[np.ix_(sel, sel)]
        sel_prim = prune_ensemble_max_imv(H_l, l, n)
        return [sel[i] for i in sel_prim]

    # Grow the size of the ensemble
    while len(sel) < n:
        
        not_sel = [i for i in range(m) if i not in sel]
        imvs = []
        for i in not_sel:
            imv = H[i, sel].sum() + H[sel, i].sum()
            imvs.append(imv)
        # print(imvs)
        sel.append(not_sel[np.argmax(imvs)])
    return sorted(sel)
    


# Not used right now
def prune_max_imv_then_kmedoids(H: np.ndarray, m: int, n: int, p: float) -> list[int]:
    """
    Select n representants from a group of models of size m. Apply max imv
    heuristic first to eliminate p percent of the overpopulation. Then select
    the n representants using medoids clustering.
    """
    assert 0 <= p <= 1, "parameter p must be a percentage"

    # Compute the intermediate matrix size and perform selection with max imv heuristic
    intermediate_size = m - int((m - n) * p)
    selected_max_imv = prune_ensemble_max_imv(H, m, intermediate_size)
    # Modify the matrix 
    H_current = H[np.ix_(selected_max_imv, selected_max_imv)]
    # Perform selection with monoid clustering
    return prune_ensemble_kmedoids(H_current, intermediate_size, n)


# Not used right now
def prune_kmedoids_then_max_imv(H: np.ndarray, m: int, n: int, p: float) -> list[int]:
    """
    Select n representants from a group of models of size m. Starts with
    kmedoids, then selects most distant ones using max imv.
    """
    assert 0 <= p <= 1, "parameter p must be a percentage"

    # Compute the intermediate matrix size and perform selection with max imv heuristic
    intermediate_size = m - int((m - n) * p)
    selected_max_imv = prune_ensemble_kmedoids(H, m, intermediate_size)
    # Modify the matrix 
    H_current = H[np.ix_(selected_max_imv, selected_max_imv)]
    # Perform selection with monoid clustering
    return prune_ensemble_max_imv(H_current, intermediate_size, n) 


def detect_outliers_imv_z(scores_flat: list[float], m: int, n: int, z_thresh: float = 2.5) -> list[int]:
    """
    Detects outliers using a z-score test, where an anomaly is defined
    as an exceptionally high IMV score of a model. 
    """
    # Compute the IMV score for each model
    H = scores_flat_to_normalized_cross_entropy(scores_flat, m)
    imv_scores = compute_imv_scores(H, m)
    # Perform a Z-test
    mean = np.mean(imv_scores)
    std = np.std(imv_scores)
    return [i for i, score in enumerate(imv_scores) if (score - mean) / std > z_thresh]


def detect_outliers_cluster(scores_flat: list[float], m: int, n: int, clusterer, matrix_maker) -> list[int]:
    """
    Detects outliers using a clustering method. Clusters of size 1 are regarded as anomalous.
    """
    M = matrix_maker(scores_flat, m)
    labels, centers = clusterer(M, m, n)

    # Clustering
    sizes: list[int] = [0 for _ in centers]
    for l in labels:
        sizes[l] += 1

    outliers = []
    for i in range(len(centers)):
        if sizes[i] == 1:
            outliers.append(centers[i])
    return outliers


def remove_outliers_from_scores_flat(scores_flat: list[float], n: int, outliers: list[int]) -> list[float]:
    # Flat array into matrix
    M = np.array(scores_flat, dtype=float).reshape((n, n))
    # Remove the outliers
    remaining = [i for i in range(n) if i not in outliers]
    M = M[np.ix_(remaining, remaining)]
    # Return flattened array
    return M.flatten().tolist()


def reindex_with_outliers(selected: list[int], outliers: list[int], n: int) -> list[int]:
    """
    Puts the selected indices back into the original indexes from 0 to n-1
    """
    non_outliers = [i for i in range(n) if i not in outliers]
    return np.array(non_outliers)[selected].tolist()


class PruningMethod:
    
    def __init__(self, name: str, pruner, matrix_maker, outlier_detector = None, custom_runner = None) -> None:
        self.name = name
        self.pruner = pruner
        self.matrix_maker = matrix_maker
        self.outlier_detector = outlier_detector
        self.custom_runner = custom_runner
    
    def prune(self, scores_flat: list[float], m: int, n: int) -> list[int]:
        if self.custom_runner != None:
            return self.custom_runner(scores_flat, m, n)
        if self.outlier_detector != None:
            return self.prune_with_outlier_detection(scores_flat, m, n)
        return self.prune_normal(scores_flat, m, n)

    def prune_normal(self, scores_flat: list[float], m: int, n: int) -> list[int]:
        if m <= n:
            selected = [i for i in range(m)]
        else:
            M = self.matrix_maker(scores_flat, m)
            selected = self.pruner(M, m, n)
        return selected

    def prune_with_outlier_detection(self, scores_flat: list[float], m: int, n: int) -> list[int]:
        assert self.outlier_detector != None, "Cannot do outlier detection with the detector = None"

        # If outlier detection is used modify the scores accordingly
        outliers = self.outlier_detector(scores_flat, m, n)
        scores_flat = remove_outliers_from_scores_flat(scores_flat, m, outliers)
        m_prim = m - len(outliers)

        # Now do the prunning
        if m_prim <= n:
            selected = [i for i in range(m_prim)]
        else:
            M = self.matrix_maker(scores_flat, m_prim)
            selected = self.pruner(M, m_prim, n)
        
        return reindex_with_outliers(selected, outliers, m)

    def DetectOutliersImvZ(self, z_thresh: float = 2.5) -> "PruningMethod":
        detector = lambda x, m, n: detect_outliers_imv_z(x, m, n, z_thresh=z_thresh)
        return PruningMethod(f"{self.name} IMV-Z", self.pruner, self.matrix_maker, detector)

    def DetectOutliersCluster(self) -> "PruningMethod":
        detector = lambda x, m, n: detect_outliers_cluster(x, m, n, cluster_ensemble_affinity, scores_flat_to_similarity_matrix_gao)
        return PruningMethod(f"{self.name} CLUSTER", self.pruner, self.matrix_maker, detector)

    @staticmethod
    def MaxImv() -> "PruningMethod":
        return PruningMethod("MAX-IMV", prune_ensemble_max_imv, scores_flat_to_normalized_cross_entropy)

    @staticmethod
    def KMedoidsGao() -> "PruningMethod":
        return PruningMethod("KMEDOIDS-GAO", prune_ensemble_kmedoids, scores_flat_to_distance_matrix_gao)

    @staticmethod
    def KMedoidsSimple() -> "PruningMethod":
        return PruningMethod("KMEDOIDS-SIMPLE", prune_ensemble_kmedoids, scores_flat_to_distance_matrix_simple)

    @staticmethod
    def Affinity() -> "PruningMethod":
        return PruningMethod("AFFINITY", prune_ensemble_affinity, scores_flat_to_similarity_matrix_gao)

    @staticmethod
    def AffinityNorm() -> "PruningMethod":

        def custom_runner(scores_flat: list[float], m: int, n: int) -> list[int]:

            S = scores_flat_to_similarity_matrix_gao(scores_flat, m)
            sel_affinity = prune_ensemble_affinity(S, m, n)
            H = scores_flat_to_normalized_cross_entropy(scores_flat, m)
            sel_final = normalize_pruned_size_with_imv(H, m, n, sel_affinity)
            return sel_final

        return PruningMethod("AFFINITY_NORM", None, None, custom_runner=custom_runner)

    # @staticmethod
    # def MaxThenMedoids(p: float) -> "PruningMethod":
    #     return PruningMethod(
    #         "MAX-IMV then KMEDOIDS", 
    #         lambda H, m, n: prune_max_imv_then_kmedoids(H, m, n, p)
    #     )

    # @staticmethod
    # def MedoidsThenMax(p: float) -> "PruningMethod":
    #     return PruningMethod(
    #         "KMEDOIDS then MAX-IMV", 
    #         lambda H, m, n: prune_kmedoids_then_max_imv(H, m, n, p)
    #     )

