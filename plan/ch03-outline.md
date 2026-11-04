# Chapter 3: Factor analysis

All three assigned transcripts read fully. Selected R video frame440 inspected:
ability.cov covariance converted with cov2cor; fa.parallel both n.obs112 iter100;
fa(...nfactors2,rotate none,fm pa); five-stock correlation matrix read.csv;
fa.parallel both iter100 without sample count; fa(...nfactors2,rotate none/varimax,
fm pa). The extra ability example is in code at the start, not an unknown data set.

1. Motivate common variation without asserting that extracted factors are real
   causes or naturally interpretable. Define x=mu+Lambda f+u, all domains,
   E(f)=E(u)=0, Cov(f)=I, Cov(u)=Psi diagonal, Cov(f,u)=0 as assumptions.
2. Derive covariance Lambda Lambda^T+Psi, common variances, unique variances,
   variable-factor covariance; contrast PCA fitting total diagonal variation.
3. Prove orthogonal rotational nonidentification and corresponding score change;
   define varimax and distinguish oblique factor correlation Phi.
4. Derive principal-axis iteration with SMC initialization and source convergence
   convention; distinguish it from PCA, MINRES parallel-analysis factors, and ML.
5. Convert ability summary and exact25-entry stock matrix examples. Unknown
   stock sample count forces source default100 null; no original scores or return
   rows available. Correct source claim that default rotation was used: displayed
   code explicitly rotate none and fm pa. No unsupported universal ranking of
   least-squares residual quality or computational speed.
6. Explain categorical indicators and why sqrt(p) modality weighting is not
   ordinary Bernoulli standard deviation or a generic normal-factor assumption.
7. Six fully solved exercises and deterministic Python numerical checks. Fresh R
   reference comparisons are available and must be recorded individually.
