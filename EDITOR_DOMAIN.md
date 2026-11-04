# Domain: predictive modelling with applications

An independent graduate study edition of Sayantee Jana's IIT Hyderabad NPTEL
course, comprising 39 teaching recordings and one orientation. The user has
authorized autonomous completion, substantive repairs, and all R-to-Python
conversions. The user does not want intermediate approval requests.

Source scope and exact assignments: plan/chapters.json and source-manifest.json.
Private timestamped text: sources/private/transcripts/VIDEO_ID.txt. Acquisition
records: sources/private/acquisition/VIDEO_ID.json. A source is not read merely
because the file exists. Chapter authors record full reading separately.

Never blur covariance PCA and correlation PCA; PCA and common-factor analysis;
linkage definitions and cluster labels; least-squares geometry and normal-theory
inference; outliers and leverage; prediction risk and parameter uncertainty;
diagnostics and model selection; odds and probability; a conditional association
and a causal cutoff estimand. State rank and sampling conditions locally.

Write chapters directly as HTML body fragments under chapters/chNN.html using
semantic section/h2/h3 and $/$$ mathematics. Every h2 and h3 has an id. Prose
does not live in scripts. The root builder will add the shared reader shell,
MathJax, navigation, code-copy buttons, and links to executed laboratories.
Each chapter needs substantial connected exposition, derived core results,
examples, six fully solved exercises, a limitation discussion, and precise
source anchors. A timestamp connects the discussion, not all authored additions.

Labs live under labs/chNN.py, with standalone execution and deterministic output
where possible. Each printed numerical result must come from a run. Each source
R example receives a distinct conversion ledger entry; do not assume all R
examples have an R upload title. Missing source data is a reproduction limitation,
not proof that the algorithm is impossible to convert. Do not claim fresh R
comparison unless both implementations were actually executed.

Source evidence, local proofs, cited results, numerical checks, source-free
rebuilds, independent scientific review, and publication checks are distinct.
No comprehensive visual audit or human certification is implied. Primary-source
references must be opened before using their claimed content or locator.

The notation contract is plan/notation.md. q=p+1 counts regression coefficients
including intercept. Technical terms need definitions at first substantive use.
The user-facing surface is dark HTML. See spec/house-style.md and WORKFLOW.md.

Reviewer outputs: rank findings and provide precise file/section locations and
replacement text. Inspect source fidelity, R conversion inventory, mathematical
claims, solutions and executable evidence. Two separate rounds are required;
review your assigned chapters only, never claim a whole-book acceptance from a
sample. State what was not verified and bind each verdict to a SHA-256 digest of
the authored chapter fragment (builder shell changes do not affect this digest).
