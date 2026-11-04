# Chapter 1: Data, prediction, and correlation

Sources read fully: aF8LvZhelzw (introductory definitions, 16–1754 s),
pi-nFW0SOV8 (correlation, 16–1389 s). No R code demonstrated in either transcript.
Do not reuse unverified retail history or the football-count illustration.

1. A prediction task needs a target and an observational unit. Define numeric
   counts, continuous measurements, categories, temporal/panel structure.
   Correct the source's conflation of discrete and qualitative variables and
   its exclusion of irrational real values. Check arbitrary category recoding.
2. Separate supervised and unsupervised objectives. Define fitted predictor,
   squared loss, and held-out observations; derive the conditional mean target.
3. Define empirical covariance with 1/n and sample covariance with 1/(n-1).
   Reproduce the source pairs x=(1,3,9,11), y=(5,9,21,25), distinguish covariance
   convention, units and pairing. Derive the computational identity.
4. Define Pearson correlation and prove its range by centered-vector
   Cauchy-Schwarz. Derive affine scaling behavior and equality conditions.
5. Reproduce x in {-3,-2,-1,1,2,3}, y=x^2; explain uncorrelated versus independent,
   association versus causation, and the limits of a single summary.
6. Python laboratory and six solved exercises. Run all numerical examples;
   no inferred R example inventory outside these two recordings.

Section claims are local proofs, definitions, or executed numerical checks.
No comprehensive visual inspection is claimed.
