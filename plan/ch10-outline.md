# Chapter 10: Categorical predictors

Full source vgvKOKD-daM read. Convert source Credit models Balance~Student,
Balance~Gender and Balance~Ethnicity; reproduce baseline swaps and -1/+1 coding.
Downloaded400-row Credit CSV, confirmed against primary CRAN ISLR Credit.rda
by fresh R extraction. Source/manual ten-thousand/default wording conflicts
with the actual Credit dataset; calculations use observed400 rows and Balance.

Sections: categories versus numerical scores; binary group means and normal
equations; reparameterization and invariance; K-category rank and reference;
mixed continuous/categorical models, contrasts and deployment schemas; exact
Python/source data conversion; six worked exercises; scope limits.

Checks: all models equal direct group-mean formulas; altered parameter bases
preserve fitted values; K indicators plus intercept are rank deficient; explicit
reference levels avoid Python/R lexical differences, including leading-space
' Male' in the source factor labels. Keep association distinct from causality.
