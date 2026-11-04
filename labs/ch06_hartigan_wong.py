"""R-compatible Hartigan--Wong k-means and seeded sampling.

Python adaptation of R's AS136 implementation (Royal Statistical Society
1979; B. D. Ripley 1998 and R Core modifications) and R Core's kmeans
wrapper, RNG initialization and Rejection sample.int routines. Distributed
under GPL-2-or-later; see ch06-hartigan-wong-LICENSE.txt.

R source: src/library/stats/src/kmns.f, stats/R/kmeans.R,
src/main/RNG.c and src/main/random.c in https://github.com/wch/r-source.
This module needs NumPy, not an R installation or saved R output.
"""
import math
import secrets
import warnings

import numpy as np

MASK = 0xffffffff
BIG = 1e30


class RRandom:
    """R set.seed + Mersenne-Twister + Rejection sample.int state.

    This preserves one stream across calls. It does not implement other
    R RNGkind/sample.kind settings or weighted sampling.
    """

    def __init__(self, seed=1):
        seed = int(seed) & MASK
        for _ in range(50):
            seed = (69069 * seed + 1) & MASK
        state = []
        for _ in range(625):
            seed = (69069 * seed + 1) & MASK
            state.append(seed)
        self.state = state[1:]
        self.position = 624

    def uniform(self):
        """One R unif_rand value, including R's endpoint fixup."""
        if self.position >= 624:
            for index in range(624):
                high = self.state[index] & 0x80000000
                low = self.state[(index + 1) % 624] & 0x7fffffff
                word = high | low
                other = self.state[(index + 397) % 624]
                self.state[index] = other ^ (word >> 1)
                if word & 1:
                    self.state[index] ^= 0x9908b0df
            self.position = 0
        word = self.state[self.position]
        self.position += 1
        word ^= word >> 11
        word ^= (word << 7) & 0x9d2c5680
        word ^= (word << 15) & 0xefc60000
        word ^= word >> 18
        value = (word & MASK) * 2.3283064365386963e-10
        if value <= 0:
            return 0.5 * 2.328306437080797e-10
        return value

    def index(self, population):
        """R's rejection-sampled integer in [0, population)."""
        if population <= 0:
            return 0
        bits = math.ceil(math.log2(population))
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + math.floor(self.uniform() * 65536)
            value &= (1 << bits) - 1
            if value < population:
                return value

    def sample_int(self, population, size, replace=False):
        """R sample.int order, returned zero-based for Python indexing."""
        population, size = int(population), int(size)
        if population < 1 or size < 0 or (not replace and size > population):
            raise ValueError("Invalid sampling population or size")
        if replace or size < 2:
            return np.array([self.index(population) for _ in range(size)])
        pool = list(range(population))
        result = []
        remaining = population
        for _ in range(size):
            index = self.index(remaining)
            result.append(pool[index])
            remaining -= 1
            pool[index] = pool[remaining]
        return np.asarray(result, dtype=int)


def _distance(point, center, stop=math.inf):
    value = 0.0
    for first, second in zip(point, center):
        difference = first - second
        value += difference * difference
        if value >= stop:
            break
    return value


class _AS136:
    """Literal optimal-transfer and quick-transfer state machine."""

    def __init__(self, data, centers):
        self.x = data
        self.c = centers.copy()
        self.m, self.p = data.shape
        self.k = len(centers)
        self.ic1 = np.zeros(self.m, dtype=int)
        self.ic2 = np.ones(self.m, dtype=int)
        self.nc = np.zeros(self.k, dtype=int)
        self.an1 = np.zeros(self.k)
        self.an2 = np.zeros(self.k)
        self.ncp = np.full(self.k, -1, dtype=int)
        self.itran = np.ones(self.k, dtype=int)
        self.live = np.zeros(self.k, dtype=int)
        self.d = np.zeros(self.m)
        self.indx = 0

    def initialize(self):
        for i, point in enumerate(self.x):
            first, second = 0, 1
            d1 = _distance(point, self.c[first])
            d2 = _distance(point, self.c[second])
            if d1 > d2:
                first, second = second, first
                d1, d2 = d2, d1
            for cluster in range(2, self.k):
                value = _distance(point, self.c[cluster], stop=d2)
                if value >= d2:
                    continue
                if value >= d1:
                    second, d2 = cluster, value
                else:
                    second, d2 = first, d1
                    first, d1 = cluster, value
            self.ic1[i], self.ic2[i] = first, second
        self.c[:] = 0.0
        for i, cluster in enumerate(self.ic1):
            self.nc[cluster] += 1
            self.c[cluster] += self.x[i]
        if np.any(self.nc == 0):
            raise ValueError("R Hartigan-Wong initial assignment is empty")
        for cluster, size in enumerate(self.nc):
            self.c[cluster] /= size
            self.an2[cluster] = size / (size + 1)
            self.an1[cluster] = size / (size - 1) if size > 1 else BIG

    def transfer(self, i, first, second):
        old_first, old_second = self.nc[first], self.nc[second]
        new_first, new_second = old_first - 1, old_second + 1
        self.c[first] = (self.c[first] * old_first - self.x[i]) / new_first
        self.c[second] = (self.c[second] * old_second + self.x[i]) / new_second
        self.nc[first], self.nc[second] = new_first, new_second
        self.an2[first] = new_first / old_first
        self.an1[first] = (new_first / (new_first - 1)
                           if new_first > 1 else BIG)
        self.an1[second] = new_second / old_second
        self.an2[second] = new_second / (new_second + 1)
        self.ic1[i], self.ic2[i] = second, first

    def optra(self):
        self.live[self.itran == 1] = self.m + 1
        for i, point in enumerate(self.x):
            step = i + 1
            self.indx += 1
            first, second = self.ic1[i], self.ic2[i]
            old_second = second
            if self.nc[first] != 1:
                if self.ncp[first] != 0:
                    self.d[i] = _distance(point, self.c[first]) * self.an1[
                        first]
                r2 = _distance(point, self.c[second]) * self.an2[second]
                for cluster in range(self.k):
                    not_live = all((step >= self.live[first],
                                    step >= self.live[cluster]))
                    if not_live or cluster in (first, old_second):
                        continue
                    threshold = r2 / self.an2[cluster]
                    distance = _distance(point, self.c[cluster], threshold)
                    if distance < threshold:
                        r2 = distance * self.an2[cluster]
                        second = cluster
                if r2 >= self.d[i]:
                    self.ic2[i] = second
                else:
                    self.indx = 0
                    self.live[first] = self.m + step
                    self.live[second] = self.m + step
                    self.ncp[first] = step
                    self.ncp[second] = step
                    self.transfer(i, first, second)
            if self.indx == self.m:
                return
        self.itran[:] = 0
        self.live -= self.m

    def qtran(self):
        count, step = 0, 0
        while True:
            for i, point in enumerate(self.x):
                count += 1
                step += 1
                if step >= 50 * self.m:
                    return False
                first, second = self.ic1[i], self.ic2[i]
                if self.nc[first] != 1:
                    if step <= self.ncp[first]:
                        self.d[i] = _distance(point, self.c[first])
                        self.d[i] *= self.an1[first]
                    if step < self.ncp[first] or step < self.ncp[second]:
                        threshold = self.d[i] / self.an2[second]
                        distance = _distance(point, self.c[second], threshold)
                        if distance < threshold:
                            count = self.indx = 0
                            self.itran[first] = self.itran[second] = 1
                            self.ncp[first] = self.ncp[second] = step + self.m
                            self.transfer(i, first, second)
                if count == self.m:
                    return True

    def run(self, iter_max):
        self.initialize()
        fault = 0
        for iteration in range(1, iter_max + 1):
            self.optra()
            if self.indx == self.m:
                break
            if not self.qtran():
                fault = 4
                break
            if self.k == 2:
                break
            self.ncp[:] = 0
        else:
            iteration = iter_max + 1
            fault = 2
        # Recompute centers and WSS in the same summation order as Fortran.
        self.c[:] = 0.0
        for i, cluster in enumerate(self.ic1):
            self.c[cluster] += self.x[i]
        within = np.zeros(self.k)
        for feature in range(self.p):
            for cluster in range(self.k):
                self.c[cluster, feature] /= self.nc[cluster]
            for i, cluster in enumerate(self.ic1):
                difference = self.x[i, feature] - self.c[cluster, feature]
                within[cluster] += difference * difference
        return {"cluster": self.ic1.copy() + 1, "centers": self.c.copy(),
                "withinss": within, "size": self.nc.copy(),
                "iter": iteration, "ifault": fault}


def kmeans_r(data, centers, nstart=1, seed=None, iter_max=10, rng=None):
    """R default kmeans: Hartigan--Wong, random rows, best of nstart.

    Passing RRandom preserves an R-compatible stream across fits. A seed
    creates a new stream; with neither, a fresh entropy-derived seed is
    used. Unseeded lecture outputs therefore cannot be reconstructed.
    Explicit initial-center matrices bypass sampling, just as in R.
    Near-tied restart totals can be ordered differently by floating-point
    summation across implementations, changing the winning restart or label
    numbering even when the partition and objective agree to tolerance.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Need a finite two-dimensional numeric matrix")
    if iter_max < 1 or nstart < 1:
        raise ValueError("iter_max and nstart must be positive")
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or an existing RNG state")
    rng = rng if rng is not None else RRandom(
        seed if seed is not None else secrets.randbits(32))
    m, p = data.shape
    numeric_centers = np.ndim(centers) == 0
    distinct = None
    if numeric_centers:
        k = int(centers)
        if k != centers or k < 1 or (k >= m and k != 1):
            raise ValueError("R Hartigan-Wong needs 1 <= k < observations")
        if nstart == 1:
            initial = data[rng.sample_int(m, k)].copy()
        if nstart >= 2 or len(np.unique(initial, axis=0)) != k:
            _, indices = np.unique(data, axis=0, return_index=True)
            distinct = data[np.sort(indices)]
            if len(distinct) < k:
                raise ValueError("More centers than distinct observations")
            initial = distinct[rng.sample_int(len(distinct), k)].copy()
    else:
        initial = np.atleast_2d(np.asarray(centers, dtype=float))
        k = len(initial)
        if initial.shape[1] != p or len(np.unique(initial, axis=0)) != k:
            raise ValueError("Initial centers must be distinct feature rows")
        if k >= m and k != 1:
            raise ValueError("R Hartigan-Wong needs fewer centers than rows")

    def one(start):
        if k == 1:
            center = data.mean(axis=0, keepdims=True)
            within = np.array([np.sum((data - center) ** 2)])
            return {"cluster": np.ones(m, dtype=int), "centers": center,
                    "withinss": within, "size": np.array([m]),
                    "iter": 1, "ifault": 0}
        return _AS136(data, start).run(iter_max)

    best = one(initial)
    best_total = best["withinss"].sum()
    if distinct is not None:
        for _ in range(1, int(nstart)):
            initial = distinct[rng.sample_int(len(distinct), k)].copy()
            candidate = one(initial)
            total = candidate["withinss"].sum()
            if total < best_total:
                best, best_total = candidate, total
    best["totss"] = np.sum((data - data.mean(axis=0)) ** 2)
    best["tot_withinss"] = best_total
    best["betweenss"] = best["totss"] - best_total
    if best["ifault"]:
        warnings.warn(f"Hartigan-Wong ifault={best['ifault']}", RuntimeWarning)
    return best


def main():
    from pathlib import Path

    import pandas as pd

    stream = RRandom(123)
    print("R set.seed(123) first five uniforms:",
          [round(stream.uniform(), 10) for _ in range(5)])
    stream = RRandom(123)
    print("R sample.int(50,4):", stream.sample_int(50, 4) + 1)
    folder = Path(__file__).resolve().parent / "data"
    wine = pd.read_csv(folder / "ch06-wine-uci.data", header=None)
    wine = wine.iloc[:, 1:].to_numpy()
    for starts in (1, 25):
        fit = kmeans_r(wine, 3, nstart=starts, seed=278613)
        print(f"Raw wine k3 nstart{starts} seed278613:")
        print("  sizes:", fit["size"])
        print("  total withinss:", fit["tot_withinss"])
    arrests = pd.read_csv(folder / "ch05-USArrests.csv", index_col=0)
    arrests = (arrests - arrests.mean()) / arrests.std(ddof=1)
    fit = kmeans_r(arrests, 4, nstart=25, seed=123)
    print("Sample-SD standardized USArrests k4 nstart25 seed123:")
    print("  withinss:", fit["withinss"])
    print("  total withinss:", fit["tot_withinss"])
    print("Unseeded source fits require a disclosed new reproducibility seed;")
    print("their original RNG state is unavailable.")


if __name__ == "__main__":
    main()
