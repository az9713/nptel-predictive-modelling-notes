# Chapter4: Clustering and linkage

Both assigned sources AZ6KZikxguU and G83gaUNketk read fully.

1. Define row clustering versus column feature reduction; partition, objective,
   dissimilarity, and no guarantee of naturally existing groups or lower cost.
2. Define metric properties; derive L1/L2 comparisons and triangle inequality;
   specify Minkowski exponent>=1, counterexample below1. Scaling is a weight.
3. Mahalanobis distance for positive-definite covariance, whitening proof,
   singular case; correlation dissimilarity is not automatically a metric.
4. Jaccard for binary presence sets and Gower for mixed observations; missing
   feature treatment must be explicit, no arbitrary label arithmetic.
5. Single/complete/average linkage formulas; centroid distinction; average-linkage
   weighted recurrence; greedy nested agglomeration versus divisive methods.
6. Six solved problems. No source R demonstration in these two theory caption
   tracks; Python lab is authored and independently checks scipy distances and
   linkage against transparent calculations.
