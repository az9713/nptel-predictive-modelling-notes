# Chapter 13 section plan

Source reading complete: C7ihAtunKro, D_oscdlZD40. Both full timestamped
caption files were read sequentially. Equations reconstructed from mathematics;
complete whiteboard/frame review was not performed.

1. Conditional error normality: define fixed design, Gaussian errors, residual
projection. Prove covariance sigma²(I-H). Counterexample: normal marginal
response is neither necessary nor sufficient for the specified conditional model.
2. Q-Q comparison: define order statistics and ppoints, explain affine normal
reference line, derive quantile-envelope scale, separate pointwise coverage
from simultaneous coverage and a diagnostic from proof of a null.
3. EDF and KS: define EDF and both one-sided jump discrepancies; describe fixed
continuous null calibration and why fitted parameters/residual dependence change it.
4. Shapiro-Wilk: derive order-statistic GLS scale direction, normalized weight
statistic, shift/scale invariance, lower-tail rejection. Treat coefficients and
p-value approximations as implemented numerical machinery, not derive them falsely.
5. Convert ToothGrowth histogram, qqnorm, qqline, car::qqPlot default envelope,
and shapiro.test. Preserve data and conventions, report actual Python output.
6. Six exercises: conditional versus marginal model; pointwise bands; EDF
calculation; projection covariance; affine-invariant check; residual diagnostic
and simulated calibration. All fully solved; numerical parts checked by lab.
