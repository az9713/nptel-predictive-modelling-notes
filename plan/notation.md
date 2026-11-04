# Book-wide notation contract

All mathematical objects are real unless a section explicitly states otherwise.
Physical units are those of the observed variable; correlations, probabilities,
leverage, odds, and test statistics are dimensionless. Local text must still
define every symbol at first use.

| Symbol | Meaning and domain | Scope |
|---|---|---|
| n | Number of observations, positive integer | book |
| p | Number of regression slopes, nonnegative integer | regression |
| q | Number of regression coefficients including intercept, q=p+1 | regression |
| d | Feature dimension for multivariate learning, positive integer | unsupervised |
| X | Regression design matrix in R^(n by q), including an intercept unless stated | regression |
| Z | Centered observed feature matrix in R^(n by d) | unsupervised |
| y | Response vector in R^n; binary entries for logistic regression | regression |
| x_i | Column vector of q predictors for row i of X | regression |
| beta | Regression coefficient vector in R^q | regression |
| epsilon | Error vector in R^n | linear regression |
| e | Residual y-X beta_hat | linear regression |
| sigma^2 | Homoscedastic conditional error variance, strictly positive | linear regression |
| Omega | Conditional error covariance, symmetric positive definite when inverted | regression |
| H | Orthogonal projection onto col(X) | linear regression |
| h_ii | Diagonal entry of H, leverage of observation i | diagnostics |
| S | Sample covariance matrix Z^T Z/(n-1), n>1 | unsupervised |
| lambda_j, v_j | Ordered covariance eigenvalue and unit eigenvector | PCA |
| k | Retained components, factors, or clusters, explicitly local to each chapter | chapter |
| Lambda | Factor-loading matrix | factor analysis |
| f, u, Psi | Latent factors, unique component, and uniqueness covariance | factor analysis |
| C_j, mu_j | Nonempty cluster j and its centroid | clustering |
| pi_i | Conditional probability P(Y_i=1 given x_i) | logistic regression |
| ell | Log likelihood as a function of beta | logistic regression |
| D | Binary treatment indicator | regression discontinuity |
| r, c | Running variable and its cutoff | regression discontinuity |
| Y(0), Y(1) | Potential outcomes without and with treatment | regression discontinuity |
| tau | Causal discontinuity estimand at the cutoff | regression discontinuity |

Overloaded chapter-local notation is always declared again; it must not migrate
into a later chapter with an unexplained changed meaning. Reserve R for the
programming language; use calligraphic risk notation when typesetting risk.
