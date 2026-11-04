# Chapter 11: Interactions and nonlinear features

Read HMpiRyz2c9c fully. Source examples: Credit Balance~Income+Student;
Balance~Income*Student and equivalent expanded formula; plots with parallel
and group-specific slopes; MASS Boston medv~lstat, scatter and standard lm
diagnostic plots; quadratic medv~lstat+I(lstat^2). All can be converted.

Use actual Credit rows from chapter10, checked primary RDA, and actual Boston
lstat/medv extracted from installed MASS. Fresh R reference fits available.
Sections: interaction means and slopes; difference-in-differences algebra;
centering with hierarchical terms; polynomial feature linearity; model comparison
and interpolation limit; source data and Python ports; six worked exercises.

Corrections: an interaction is a conditional mean feature, not correlation
between predictors; adjusted R² increase is not validated prediction improvement;
n distinct input values interpolate with degree at most n-1, not necessarily n.
Polynomial extrapolation and historical aggregate data receive explicit limits.
