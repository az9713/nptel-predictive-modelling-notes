# Book specification — autonomous study edition

## Working title and conceptual spine

Predictive Modelling with Applications: Supervised and Unsupervised Learning.
An independent study edition of Prof. Sayantee Jana's IIT Hyderabad course,
hosted by NPTEL. The local directory name and the uploader's IITM label do not
identify the teaching institution.

The spine is the transition from describing multivariate variation to fitting
conditional models, diagnosing their failures, assessing new-data prediction,
and stating the additional assumptions needed for causal identification.

## Reader and prerequisites

A quantitatively trained graduate beginning this subject. Assume real vector
spaces, matrix multiplication, eigenvalues and the spectral theorem, partial
derivatives, elementary optimization, expectation, covariance, and elementary
probability distributions. Supply a shared notation reference and introduce
subject-specific statistical concepts locally.

## Reader before and after

Before: can calculate a covariance matrix and solve a least-squares problem,
but has not necessarily studied model diagnostics or multivariate learning.

After: can derive PCA's variance and reconstruction formulations; distinguish
PCA from an identifiable latent-factor model; implement and critique hierarchical
and k-means clustering; derive and fit linear and logistic regression; diagnose
influence, dependence, nonnormality, collinearity, and unequal error variance;
assess a complete fitted pipeline on held-out data; state why a cutoff jump is
causal only under additional identification assumptions. Each outcome maps to
the corresponding chapter result nodes in plan/chapters.json.

## Scope and exclusions

Cover all 39 teaching recordings, with the additional introduction mapped in the
preface. Group them into 18 chapters. Preserve all original video IDs, titles,
durations, and playlist positions. Teaching order was verified against the
official course lesson array for all 39 teaching video IDs. Full assigned
caption reading is recorded separately from selected visual checks.

Convert every R-based course demonstration to Python, as explicitly requested.
Preserve the same data, preprocessing, model, and statistical interpretation.
Record package defaults, estimator conventions, randomness, and tolerances in
an example conversion ledger. If a conversion cannot be made faithfully,
explain the specific obstacle and distinguish it from a source-data retrieval
failure. Any replacement synthetic data must be labeled original and must not
impersonate a course dataset. R is not required to run the final reader or labs.

Exclude reconstruction of unavailable source material. Deep learning, language
models, full time-series forecasting, longitudinal models, and a general causal
inference course are outside scope. Explain autocorrelation as a regression
diagnostic without silently expanding into a time-series textbook. Modern
extensions such as leakage-safe preprocessing and robust standard errors are
explicitly authored additions, not attributed to the lecture without evidence.

## Results assumed without proof

The real symmetric spectral theorem, elementary differentiation, and elementary
normal/chi-square/t/F distribution facts may be cited as prerequisites. Prove
the statistical derivations that depend on them. Asymptotic diagnostic and
covariance results require explicit conditions and verified references; finite
simulation alone cannot establish them.

## Length, exercises, and laboratories

The initial budget was 216 manuscript pages, adjusted by mathematical need after
source reading. The completed bodies contain approximately 44,000 words,
108 worked solutions and 18 laboratories. Conceptual, derivational and
computational work is balanced according to each topic; some source-data
reproduction exercises remain conditional on unavailable original inputs.

## Format and numbering

Dark-mode HTML with one file per chapter; bundled local MathJax, navigation,
search, code-copy buttons, expandable solutions, accessible mobile layouts,
and print styles that expose solutions. Number equations, results, and figures
by chapter. Add source map, study guide, notation reference, lab index, and
separate provenance and validation records. A clean export must build without
private transcripts, downloaded media, or a network connection.

## Primary sources and decisions reserved

The supplied YouTube playlist and official course page govern coverage and
identity. Read source text in full before writing each chapter; inspect frames
for ambiguous equations and confirm important external results in primary
references. An acquisition or a source timestamp is not evidence of complete
reading or visual verification.

The user authorized completion, repairs and independent reviews autonomously.
Public repository creation, pushing, and Pages publication have not been
requested. Keep all project artifacts in this folder.
