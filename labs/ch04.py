"""Transparent distance and linkage calculations for the clustering theory."""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, pdist, squareform


def gower_pair(left, right, kinds, ranges, weights=None):
    """Complete-data Gower variant: numeric and nominal features."""
    if weights is None:
        weights = np.ones(len(kinds))
    contributions = []
    for a, b, kind, span in zip(left, right, kinds, ranges):
        if kind == "numeric":
            if span <= 0:
                raise ValueError("Numeric range must be strictly positive")
            contributions.append(abs(float(a) - float(b)) / span)
        elif kind == "nominal":
            contributions.append(float(a != b))
        else:
            raise ValueError("Unknown feature type")
    return np.average(contributions, weights=weights)


def agglomerate(points, method):
    """Small direct implementation of single/complete/size-weighted average."""
    distance = squareform(pdist(points))
    clusters = {i: [i] for i in range(len(points))}
    result = []
    while len(clusters) > 1:
        candidates = []
        identifiers = sorted(clusters)
        for position, left in enumerate(identifiers):
            for right in identifiers[position + 1:]:
                cross = distance[np.ix_(clusters[left], clusters[right])]
                criterion = {
                    "single": np.min,
                    "complete": np.max,
                    "average": np.mean,
                }[method](cross)
                candidates.append((criterion, left, right))
        height, left, right = min(candidates)
        members = clusters.pop(left) + clusters.pop(right)
        result.append([left, right, height, len(members)])
        clusters[len(points) + len(result) - 1] = members
    return np.array(result)


def main():
    np.set_printoptions(precision=6, suppress=True)
    difference = np.array([3.0, 4.0])
    print(
        "Difference(3,4): L1=",
        np.linalg.norm(difference, 1),
        "L2=",
        np.linalg.norm(difference),
    )
    assert np.linalg.norm(difference) <= np.linalg.norm(difference, 1)
    left = np.array([[0.0, 0.0], [0.0, 2.0]])
    right = np.array([[3.0, 0.0], [3.0, 2.0]])
    cross = cdist(left, right)
    print("Cross-cluster Euclidean distances:")
    print(cross)
    print(f"Single linkage: {cross.min():.6f}")
    print(f"Complete linkage: {cross.max():.6f}")
    print(f"Average linkage: {cross.mean():.6f}")
    print("Centroid distance:", np.linalg.norm(left.mean(0) - right.mean(0)))
    points = np.array([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0], [4.0, 2.0]])
    for method in ["single", "complete", "average"]:
        direct = agglomerate(points, method)
        reference = linkage(points, method=method)
        assert np.allclose(direct[:, 2:], reference[:, 2:])
        print(method, "merge heights:", direct[:, 2])
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    a, b = np.array([2.0, 1.0]), np.array([-1.0, 3.0])
    delta = a - b
    mahalanobis = np.sqrt(delta @ np.linalg.solve(covariance, delta))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = (
        eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
    )
    assert np.allclose(mahalanobis, np.linalg.norm(whitening @ delta))
    print(f"Mahalanobis versus whitened Euclidean: {mahalanobis:.6f}")
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    w = (u + v) / np.sqrt(2)
    direct = 1 - np.dot(u, v)
    indirect = (1 - np.dot(u, w)) + (1 - np.dot(w, v))
    assert direct > indirect
    print(
        f"1-correlation triangle: direct={direct:.6f}, "
        f"via third={indirect:.6f} (fails)"
    )
    binary_a = np.array([1, 0, 1, 0], dtype=bool)
    binary_b = np.array([1, 1, 0, 0], dtype=bool)
    unequal = np.logical_xor(binary_a, binary_b).sum()
    present = np.logical_or(binary_a, binary_b).sum()
    jaccard = unequal / present
    print(f"Jaccard presence distance: {jaccard:.6f}")
    mixed = gower_pair(
        [10, "A"], [30, "B"], ["numeric", "nominal"], [40, None]
    )
    print(f"Mixed numeric/nominal Gower pair: {mixed:.6f}")
    assert np.isclose(mixed, 0.75)
    exponent = 0.5
    diagonal = np.sum(np.abs(np.array([1.0, 1.0])) ** exponent) ** (
        1 / exponent
    )
    via = 2.0
    assert diagonal > via
    print(
        f"Minkowski exponent0.5: direct={diagonal:.6f}, "
        f"axis route={via:.6f} (fails triangle)"
    )
    print(
        "Checks passed: distances, linkage, whitening, counterexamples."
    )


if __name__ == "__main__":
    main()
