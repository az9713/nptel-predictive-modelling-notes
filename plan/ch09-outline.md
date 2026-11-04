# Chapter 9: Outliers, leverage and influence

Read all three assigned transcripts in full: 3EX388b-7s4, Cn572X5kKKA,
la5PiJboaoo. Source demonstration uses robustbase::delivery (25 observations),
lm(delTime~.), and olsrr (ASR says OSR) plots of standardized/studentized
residuals, leverage, studentized fits, Cook distance in bar/chart forms,
DFBETAS and DFFITS. All methods receive Python equivalents and direct-refit checks.

Sections: diagnostic distinctions; hat-matrix bounds and extremes; residual
scaling; deletion identities; Cook distance/DFFITS/DFBETAS; real delivery data
conversion and reproducible plots; six solved exercises; domain-sensitive action.

Correct source ambiguities explicitly: h_ii=0/1 concerns one row, not necessarily
the whole H; average leverage q/n requires no equal-leverage assumption; h_ii²
does not equal h_ii for a general projection diagonal; internal residuals do not
have an exact ordinary t law. Use external studentization for the exact t result.
Cutoffs are screens, not formal multivariate declarations or automatic deletion.
