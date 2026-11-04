# Required R-to-Python conversion

The user explicitly requested conversion of every R example and an explanation
for each example that cannot be converted. The final teaching code uses Python.

Inventory examples while reading recordings and inspecting code/output frames;
do not assume only upload titles mentioning R contain R demonstrations. For each
example record video ID, time range, source function/package, data origin and
hash if available, preprocessing, estimator, options, output, Python equivalent,
verification method, numerical tolerance, status, and unresolved discrepancy.

Statuses: discovered; source_incomplete; awaiting_data; implemented_unverified;
verified_against_source_output; verified_against_independent_calculation;
verified_against_fresh_r_run; blocked_with_explanation. The ledger starts empty
because the full source has not yet been read. An empty ledger is not evidence
that there are zero R examples.

Check explicitly when relevant:

- Sample versus population covariance; centering and scaling before PCA;
  singular-vector signs and rotations within repeated-eigenvalue eigenspaces.
- Factor extraction method, correlation matrix, rotation, rotation normalization,
  loading orientation, factor-score convention, and latent-sign indeterminacy.
- Hierarchical linkage definitions; squared versus unsquared distances;
  R's ward.D versus ward.D2; tie handling and cluster-label permutation.
- k-means initialization algorithm, restarts, stopping rule, random generator,
  cluster labels, and objective. A shared integer seed across languages does
  not guarantee the same random initialization.
- Formula expansion, intercept, reference category and contrasts, interaction
  columns, missing-value rows, and rank-deficient designs.
- OLS versus weighted fits; residual degrees of freedom; studentization,
  influence formulas, robust-covariance convention, and test variant.
- Normality and dependence test assumptions, approximation, and p-value method.
- Binomial family, link, weights, treatment of counts/trials, likelihood scale,
  optimizer, regularization defaults, and separation in logistic regression.
- Stepwise search direction, candidate scope, AIC conventions, and selection
  instability; distinguish reproduction from a recommended validation protocol.

No method or data obstacle has yet been demonstrated. Do not preemptively call
an example impossible. A proprietary/missing dataset can prevent exact numerical
reproduction even when the algorithm is convertible; state both facts. If a
Python implementation needs a maintained specialist package or an explicit
implementation, verify its documentation and licensing before choosing it.
