# Chapter 2: Principal components

Read all three transcripts: Zq7WPJprL8o, RbPOCNjJNlE, BwZuubr6_-c.

1. Task and centered n-by-d matrix Z; rotation alone does not reduce dimension.
2. Symmetric positive-semidefinite covariance S=Z^T Z/(n-1); spectral theorem,
   trace and variance preservation. Distinguish eigenvalues from matrix entries.
3. Prove the leading eigenvector solves a maximum-variance problem and successive
   orthogonal directions diagonalize score covariance.
4. Derive orthogonal-projection reconstruction error and its discarded-eigenvalue
   sum. Explain tied eigenspaces and sign ambiguity, and counterexample to a
   prediction guarantee for retained high-variance directions.
5. Centering versus standardizing; covariance and correlation PCA. Explain
   eigenvectors, loading correlations, raw and standardized score weights.
6. Convert source USJudgeRatings (CONT omitted deliberately, leaving 11 ratings),
   psych::principal, fa.parallel 100-iteration Gaussian/column-bootstrap nulls,
   Harman23.cor two-component none/varimax, score extraction, and prcomp(DILG,
   DECI). Correct proportions within retained variance versus original variance.
   Harman supplies only a correlation matrix and n.obs=305: individual score
   extraction is unavailable without individual measurements.
7. Six solved exercises and executable checks. Frame inspection supplements
   captions for exact function arguments; no complete audiovisual audit.
