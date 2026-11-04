"""Faithful Python port of the course's 26-index NbClust workflow.

Scope: NbClust 3.0.1, Euclidean average linkage or Hartigan--Wong
k-means, index='all', k>=2.
This is a source-behavior port, including documented package quirks;
it is not a replacement definition of the named statistical indices.
Derived from Malika Charrad, Nadia Ghazzali, Veronique Boiteau and
Azam Niknafs, NbClust R/NbClust.R, GPL-2; see adjacent license notice.
https://github.com/cran/NbClust/blob/master/R/NbClust.R
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import f

NAMES = [
    "KL", "CH", "Hartigan", "CCC", "Scott", "Marriot", "TrCovW",
    "TraceW", "Friedman", "Rubin", "Cindex", "DB", "Silhouette",
    "Duda", "Pseudot2", "Beale", "Ratkowsky", "Ball", "Ptbiserial",
    "Frey", "McClain", "Dunn", "Hubert", "SDindex", "Dindex", "SDbw",
]
BEST_NAMES = ["PseudoT2" if n == "Pseudot2" else
              "PtBiserial" if n == "Ptbiserial" else n for n in NAMES]


def _div(a, b):
    return np.divide(np.float64(a), np.float64(b))


def _centers(x, labels):
    return np.array([x[labels == j].mean(0)
                     for j in range(1, labels.max() + 1)])


def _wss(x, labels):
    centers = _centers(x, labels)
    return np.sum((x - centers[labels - 1]) ** 2)


def _r_dropped_wss(x):
    # R's x[mask, ] drops a singleton row into a vector. In Indices.WKWL,
    # as.matrix then interprets it as p rows and one column. Preserve it.
    if len(x) == 1:
        return np.sum((x[0] - x[0].mean()) ** 2)
    return np.sum((x - x.mean(0)) ** 2)


def _split_indices(x, a, b):
    first, last = 1, b.max()
    while np.sum(a == first) == np.sum(b == first) and first <= a.max():
        first += 1
    previous = last - 1
    while np.sum(a == previous) == np.sum(b == last) and previous >= 1:
        last -= 1
        previous -= 1
    nk, nl, nm = np.sum(b == first), np.sum(b == last), np.sum(a == first)
    wk = _r_dropped_wss(x[b == first])
    wl = _r_dropped_wss(x[b == last])
    wm = _r_dropped_wss(x[a == first])
    within = wk + wl
    between = wm - within
    duda = _div(within, wm)
    pseudo = _div(between, _div(within, nk + nl - 2))
    beale = _div(_div(between, within),
                 _div(nm - 1, nm - 2) * (2 ** (2 / x.shape[1]) - 1))
    variance = 2 * (1 - 8 / (np.pi ** 2 * x.shape[1])) / (nm * x.shape[1])
    z = 3.2 * np.sqrt(variance)
    critical = 1 - 2 / (np.pi * x.shape[1]) - z
    return ([duda, pseudo, beale],
            [critical, _div(1 - critical, critical) * (nk + nl - 2),
             f.sf(beale, x.shape[1], (nm - 2) * x.shape[1])])


def _wbt(x, labels, total, spectrum):
    n, p = x.shape
    k = labels.max()
    centers = _centers(x, labels)
    sizes = np.bincount(labels)[1:]
    between = centers.T @ (sizes[:, None] * centers)
    within = total - between
    trace = np.trace(within)
    determinant = np.linalg.det(within)
    marriot = k ** 2 * determinant
    trcovw = np.trace(np.cov(within, rowvar=False, ddof=1))
    scott = n * np.log(np.linalg.det(total) / determinant)
    inverse_diagonal = np.diag(np.linalg.inv(within))
    friedman = np.sum(inverse_diagonal * np.diag(between))
    rubin = np.trace(total) / trace
    ratio = 1 - trace / np.trace(total)
    scale = (np.prod(spectrum) / k) ** (1 / p)
    u = spectrum / scale
    p1 = min(np.sum(u >= 1), k - 1)
    if 0 < p1 < p:
        scale = (np.prod(spectrum[:p1]) / k) ** (1 / p1)
        u = spectrum / scale
        b1 = np.sum(1 / (n + u[:p1]))
        b2 = np.sum(u[p1:] ** 2 / (n + u[p1:]))
        expected = 1 - (b1 + b2) / np.sum(u ** 2) * (n - k) ** 2 / n * (
            1 + 4 / n)
        dimension = p1
    else:
        b1 = np.sum(1 / (n + u))
        expected = 1 - b1 / np.sum(u ** 2) * (n - k) ** 2 / n * (1 + 4 / n)
        dimension = p
    logarithm = np.log((1 - expected) / (1 - ratio))
    ccc = logarithm * np.sqrt(n * dimension / 2) / (0.001 + expected) ** 1.2
    return ccc, scott, marriot, trcovw, trace, friedman, rubin


def _separation(x, labels):
    distances = squareform(pdist(_centers(x, labels)))
    pairs = distances[np.triu_indices(len(distances), 1)]
    reciprocal_sum = np.sum(1 / distances.sum(axis=1))
    return pairs.max() / pairs.min() * reciprocal_sum


def _scatter(x, labels):
    k = labels.max()
    centers = _centers(x, labels)
    variances = np.array([
        np.mean((x[labels == j] - centers[j - 1]) ** 2, axis=0)
        for j in range(1, k + 1)])
    summed = np.linalg.norm(variances, axis=1).sum()
    stdev = np.sqrt(summed) / k
    scatter = summed / np.linalg.norm(np.var(x, axis=0)) / k
    distance = np.linalg.norm(x - centers[labels - 1], axis=1)
    density = np.array([np.sum(distance[labels == j] <= stdev)
                        for j in range(1, k + 1)])
    between_density = 0.0
    for i in range(k):
        for j in range(k):
            if i == j or max(density[i], density[j]) == 0:
                continue
            subset = x[(labels == i + 1) | (labels == j + 1)]
            midpoint = (centers[i] + centers[j]) / 2
            count = np.sum(np.linalg.norm(subset - midpoint, axis=1) <= stdev)
            between_density += count / max(density[i], density[j])
    return scatter, between_density / (k * (k - 1)), distance.mean()


def _pair_stats(distances, labels):
    upper = np.triu_indices(len(labels), 1)
    same = labels[upper[0]] == labels[upper[1]]
    values = distances[upper]
    return values[same], values[~same]


def _distance_indices(x, labels, next_labels, distances):
    n = len(x)
    k = labels.max()
    centers = _centers(x, labels)
    within, between = _pair_stats(distances, labels)
    next_within, next_between = _pair_stats(distances, next_labels)
    frey = _div(next_between.mean() - between.mean(),
                next_within.mean() - within.mean())
    mcclain = _div(within.mean() / len(within),
                   between.mean() / len(between))
    probability = len(between) / (len(within) + len(between))
    mean_difference = between.mean() - within.mean()
    factor = np.sqrt(probability * (1 - probability))
    point = mean_difference * factor / np.std(np.r_[within, between], ddof=1)
    minima, maxima, du = [], [], 0.0
    scatter = []
    silhouette = 0.0
    for j in range(1, k + 1):
        indices = np.where(labels == j)[0]
        block = distances[np.ix_(indices, indices)]
        if len(indices) == 1:
            minima.append(1.0)
            maxima.append(1.0)
            silhouette += 1
        else:
            du += block.sum() / 2
            maxima.append(block.max())
            minima.append(block[block != 0].min()
                          if np.sum(block == 0) == len(indices) else 0.0)
            for row in indices:
                ai = distances[row, indices].sum() / (len(indices) - 1)
                bi = min(distances[row, labels == other].mean()
                         for other in range(1, k + 1) if other != j)
                silhouette += _div(bi - ai, max(ai, bi))
        scatter.append(np.sqrt(np.mean(np.sum(
            (x[indices] - centers[j - 1]) ** 2, axis=1))))
    low, high = min(minima), max(maxima)
    cindex = _div(du - len(within) * low, len(within) * (high - low))
    center_distances = squareform(pdist(centers))
    numerator = np.add.outer(scatter, scatter)
    ratios = np.divide(numerator, center_distances)
    db = np.mean([np.max(row[np.isfinite(row)]) for row in ratios])
    dunn = between.min() / within.max()
    qdist = cdist(centers[labels - 1], centers[labels - 1])
    centered_p = distances - distances.mean()
    centered_q = qdist - qdist.mean()
    upper = np.triu_indices(n, 1)
    norm_p = np.linalg.norm(np.var(distances, axis=0))
    norm_q = np.linalg.norm(np.var(qdist, axis=0))
    numerator = np.sum((centered_p * centered_q)[upper])
    hubert = numerator / (n * (n - 1) / 2 * norm_p * norm_q)
    return cindex, db, silhouette / n, point, frey, mcclain, dunn, hubert


def _best(indices, critical, alpha_beale):
    ks = indices.index.to_numpy()
    best = pd.DataFrame(np.nan, index=["Number_clusters", "Value_Index"],
                        columns=BEST_NAMES)

    def extreme(name, values, maximize=True):
        values = np.asarray(values, dtype=float)
        if np.isnan(values).all():
            return
        index = np.nanargmax(values) if maximize else np.nanargmin(values)
        column = BEST_NAMES[NAMES.index(name)]
        best.loc[:, column] = [ks[index], values[index]]

    for name in ["KL", "CH", "CCC", "Silhouette", "Ptbiserial",
                 "Ratkowsky", "Dunn"]:
        extreme(name, indices[name])
    for name in ["DB", "Cindex", "McClain", "SDindex", "SDbw"]:
        extreme(name, indices[name], False)
    for name in ["Hartigan", "Scott", "TrCovW", "Friedman", "Ball"]:
        difference = np.r_[np.nan, np.abs(np.diff(indices[name]))]
        extreme(name, difference)
    for name in ["Marriot", "TraceW", "Rubin"]:
        difference = np.r_[np.nan, np.diff(indices[name], n=2), np.nan]
        extreme(name, difference, name != "Rubin")
    for name, column, comparison in [
        ("Duda", "CritValue_Duda", "ge"),
        ("Pseudot2", "CritValue_PseudoT2", "le"),
        ("Beale", "Fvalue_Beale", "p"),
    ]:
        if comparison == "p":
            good = critical[column].to_numpy() >= alpha_beale
        elif comparison == "ge":
            good = indices[name].to_numpy() >= critical[column].to_numpy()
        else:
            good = indices[name].to_numpy() <= critical[column].to_numpy()
        matches = np.where(good)[0]
        if matches.size:
            row = matches[0]
            best.loc[:, BEST_NAMES[NAMES.index(name)]] = [
                ks[row], indices[name].iloc[row]]
    matches = np.where(indices["Frey"].to_numpy() < 1)[0]
    if matches.size:
        row = matches[0]
        value = indices["Frey"].iloc[row - 1] if row else np.nan
        best.loc[:, "Frey"] = [ks[row] - 1, value]
    for name in ("Hubert", "Dindex"):
        best.loc[:, name] = [0.0, 0.0]
    counts = Counter(int(k) for k in best.loc["Number_clusters"]
                     if np.isfinite(k) and ks[0] <= k <= ks[-1])
    maximum = max(counts.values())
    tied = sorted(k for k, count in counts.items() if count == maximum)
    return best.round(4), dict(sorted(counts.items())), tied


def nbclust_average(data, min_nc=2, max_nc=15, standardize=True,
                    alpha_beale=0.1, method="average", seed=None):
    """Port the actual 26-index Euclidean course calls.

    Returns the rounded R-style tables, votes, tied pluralities, selected
    smallest tied k, and partition. Raw values are also returned for audits.
    The historical name is retained for existing chapter imports. The R
    k-means method resets set.seed(1) before EACH partition; an outer seed
    argument therefore does not control its starts. Its default is nstart1,
    iter.max10, Hartigan--Wong. Its partitions are not necessarily nested.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    n, p = x.shape
    if not 2 <= min_nc < max_nc < n or max_nc - min_nc < 2:
        raise ValueError("Need 2 <= min_nc, max_nc < n, and a range >=2")
    if standardize:
        scale = x.std(axis=0, ddof=1)
        if np.any(scale == 0):
            raise ValueError("Cannot standardize a constant feature")
        x = (x - x.mean(axis=0)) / scale
    distances = squareform(pdist(x))
    if method == "average":
        hierarchy = linkage(pdist(x), method="average")
        partitions = {k: cut_tree(hierarchy, n_clusters=k).ravel() + 1
                      for k in range(min_nc - 1, max_nc + 2)}
    elif method == "kmeans":
        from ch06_hartigan_wong import kmeans_r

        partitions = {1: np.ones(n, dtype=int)}
        for k in range(max(2, min_nc - 1), max_nc + 2):
            partitions[k] = kmeans_r(x, k, seed=1)["cluster"]
    else:
        raise ValueError("Only course-used average and kmeans are ported")
    total = x.T @ x
    spectrum = np.sqrt(np.linalg.eigvalsh(total / (n - 1))[::-1])
    spectrum[spectrum == 0] = 1
    indices = pd.DataFrame(index=range(min_nc, max_nc + 1), columns=NAMES,
                           dtype=float)
    critical = pd.DataFrame(index=indices.index, columns=[
        "CritValue_Duda", "CritValue_PseudoT2", "Fvalue_Beale"],
        dtype=float)
    tss = np.sum((x - x.mean(0)) ** 2)
    coordinate_tss = np.sum((x - x.mean(0)) ** 2, axis=0)
    alpha = _separation(x, partitions[max_nc])
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in indices.index:
            a, b, c = (partitions[k - 1], partitions[k], partitions[k + 1])
            w0, w1, w2 = (_wss(x, lab) for lab in (a, b, c))
            indices.loc[k, "KL"] = _div(
                abs((k - 1) ** (2 / p) * w0 - k ** (2 / p) * w1),
                abs(k ** (2 / p) * w1 - (k + 1) ** (2 / p) * w2))
            indices.loc[k, "CH"] = _div(
                (tss - w1) / (k - 1), w1 / (n - k))
            indices.loc[k, "Hartigan"] = (w1 / w2 - 1) * (n - k)
            indices.loc[k, NAMES[3:10]] = _wbt(x, b, total, spectrum)
            split, thresholds = _split_indices(x, b, c)
            indices.loc[k, ["Duda", "Pseudot2", "Beale"]] = split
            critical.loc[k] = thresholds
            centers = _centers(x, b)
            coordinate_within = np.sum((x - centers[b - 1]) ** 2, axis=0)
            indices.loc[k, "Ratkowsky"] = np.mean(np.sqrt(
                (coordinate_tss - coordinate_within) / coordinate_tss))
            indices.loc[k, "Ratkowsky"] /= np.sqrt(k)
            indices.loc[k, "Ball"] = w1 / k
            names = ["Cindex", "DB", "Silhouette", "Ptbiserial",
                     "Frey", "McClain", "Dunn", "Hubert"]
            indices.loc[k, names] = _distance_indices(x, b, c, distances)
            scatter, density, dindex = _scatter(x, b)
            indices.loc[k, "SDindex"] = alpha * scatter + _separation(x, b)
            indices.loc[k, "Dindex"] = dindex
            indices.loc[k, "SDbw"] = scatter + density
    best, votes, tied = _best(indices, critical, alpha_beale)
    selected = tied[0]
    return {"All.index": indices.round(4),
            "All.CriticalValues": critical.round(4), "Best.nc": best,
            "Best.partition": partitions[selected], "votes": votes,
            "tied_pluralities": tied, "selected_k": selected,
            "raw_indices": indices, "raw_critical": critical}


# More general spelling for the two supported methods; old imports still work.
nbclust = nbclust_average


def main():
    path = Path(__file__).resolve().parent / "data/ch05-nutrient.csv"
    data = pd.read_csv(path, index_col=0)
    result = nbclust_average(data)
    print("NbClust 3.0.1 course workflow: 26 indices, k=2..15")
    print(result["Best.nc"].to_string())
    print("Votes:", result["votes"])
    print("Tied pluralities:", result["tied_pluralities"])
    print("Source smallest-tie rule selects:", result["selected_k"])
    print("This is not a unique majority or a calibrated probability.")


if __name__ == "__main__":
    main()
