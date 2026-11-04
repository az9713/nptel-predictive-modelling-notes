# Chapter 7: Simple linear regression

Sources fully read: znWxD_73rtg and j35ZjlaXpPs, timestamped caption files.
Source-specific course R example: insurance CSV, male premium regressed on age,
second recording 1232–1530 seconds. Data table/frame recovery is being checked.

1. Paired observations and prediction: define conditional mean, distinguish an
   observational slope from causation, preserve row pairs, inspect scatter.
2. Least squares: exact normal equations, centered sums, unique minimizer when
   Sxx>0, residual identities as consequences rather than sampling assumptions.
3. Geometry, units, and correlation: derive slope=r sy/sx and R²=r² with intercept;
   negative association valid, correlation testing not a mandatory gate.
4. Uncertainty (authored extension): fixed design, mean-zero homoscedastic errors,
   slope variance, Gaussian t inference, mean and new-observation intervals.
5. Python conversion of read.csv/lm and insurance example: source-data limits;
   exact numerical original teaching construction with independent checks.
6. Six worked exercises: two concepts, two derivations, two computation/design.
7. Scope and source boundary: regression can handle dependent data with suitable
   assumptions; source sum-of-errors statements do not establish exogeneity.

Core checks: least-squares coefficient equals centered formula; residual
orthogonality; interval ordering; no-intercept identity failure; paired-row
permutation invariance. All printed numbers must match lab execution.
