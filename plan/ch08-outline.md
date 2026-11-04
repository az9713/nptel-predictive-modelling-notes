# Chapter 8: Multiple regression and projection geometry

Fully read BSk7RaB4kbk, dK7LyVwHlyo, rLXNVsz9qwo. Core sequence:
1. Several predictors and conditional coefficient interpretation.
2. Matrix least squares, rank, normal equations, QR/SVD, projection proof.
3. Conditional unbiasedness and covariance under explicit assumptions.
4. Residual projection, covariance versus expected squared norm, degrees of freedom.
5. Gaussian likelihood: identical coefficient MLE but SSE/n variance MLE;
   exact t and nested F with n-q residual degrees of freedom.
6. Source R demonstrations: Toyota29-row training20/validation9 with simple,
   age-adjusted, and all-predictor fits; NYC housing exploration, factor model,
   coefficient extraction/plot, new-row predictions and intervals. Convert each.
7. Six solved exercises and source/implementation limits.

Source corrections: E(epsilon)=0 does not imply sample sum epsilon=0; e includes
epsilon, not only X(beta-beta_hat). Cov(e) is sigma²(I-H), whereas E(e'e)
is scalar sigma²(n-q). Use q=p+1 when counting intercept. Exact coefficient t
uses n-q df, not n-1 or automatically n-2 in multiple regression. A nonrejected
test does not establish a zero coefficient or a causal absence of influence.

Recovered canonical author NYC housing CSVs from jaredlander.com; compare source
frames before claiming source-specific exact output. Toyota subset/seed not given
in captions, so numerical reproduction must retain that limitation.
