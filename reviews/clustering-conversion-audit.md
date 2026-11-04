# Independent scientific audit of clustering conversions

Date: 2026-11-04. Reviewer: review_01_06_r2, independent of helper and chapter authorship.

Verdict: acceptable for the explicitly documented course workflows and finite reference fixtures. This is a mathematical/code audit with bounded execution checks, not a proof of arbitrary-input equivalence or a comprehensive audiovisual inventory.

## Scope and primary implementation evidence

Read the complete Python implementations `labs/ch05_nbclust.py`, `labs/ch05_diana.py`, `labs/ch05_clusterlearn.py`, `labs/ch06_hartigan_wong.py`, and main labs 5–6. Examined the retained primary NbClust implementation, R's kmeans wrapper, AS136 optimal/quick transfer code, RNG initialization/endpoint behavior, and uniform sampling code. Independently printed the installed clustlearn 1.0.0 divisive and kmeans functions with R 4.5.1. Read fresh-reference scripts, saved comparisons, package/data provenance and adjacent GPL/MIT notices. No Python course execution imports an R runtime or substitutes fitted R outputs for its calculations. Private reference arrays are used only for cross-language validation; bundled original numerical datasets are legitimate inputs.

Selected source frames independently inspected: factor-440-detail, pca-1575 and kmeans-1347. They establish PA none/varimax and ability n.obs=112, Harman source n.obs=302, and the negative powers in the jump formula respectively. The source transcript passages checked here are targeted checks, not a second complete reading of all recordings. The manuscript authors' complete-caption reading records are separate evidence.

## NbClust 3.0.1

The helper implements the actual default `index="all"` set of 26 criteria, not an elbow/silhouette replacement. The average-linkage partitions and the source's kmeans orchestration are present. In kmeans mode the wrapper resets seed 1 before each partition and retains default nstart=1/iter.max=10; the outer seed argument deliberately does not control those source partitions. Duda-family calculations on nonnested kmeans partitions preserve the package's bookkeeping rather than claiming a genuine nested split.

Inspected the criteria, recommendation logic, derivative-based selections and smallest-tied-plurality rule against the primary R source. Several formulas intentionally reproduce package behavior rather than supply canonical statistical definitions: singleton silhouette contributes one; the Hartigan index uses n-k here, unlike the separate useful wrapper's n-k-1; C-index uses source cluster extrema; Friedman uses the diagonal of elementwise inverse-within times between; singleton-row dropping affects Duda-family WSS; Hubert/Dindex are graph-only zero recommendations; and Frey's boundary/sentinel handling is retained. These quirks must remain scoped to this port.

Independently recomputed all rounded tables from the bundled nutrient input and the independent 36-by-3 fixture, using the reference's actual candidate ranges. Every All.index, All.CriticalValues and Best.nc entry agrees exactly, including missing-value masks. Nutrient votes are `{2:4,3:4,4:2,5:4,9:1,10:1,13:2,14:1,15:4}`: the largest count is tied at 2,3,5,15; the source's selected 2 is not a unique majority. The saved standardized-wine comparison separately covers all 26 kmeans-mode columns and partition. Marriot's large magnitude, approximately 1e25, makes an absolute difference alone misleading; its documented relative discrepancy is about 1e-14.

This API is deliberately narrower than arbitrary NbClust: Euclidean average/kmeans, k>=2, a range spanning at least two steps, and nondegenerate complete numerical inputs after its filtering/scaling. It does not implement alllong, Gap, every distance/linkage option, singular-matrix fallbacks or exhaustive invalid-input compatibility. Row filtering treats all nonfinite inputs as unavailable; R's na.omit semantics for infinities are different. These are outside the finite course fixtures and must not be advertised as universal package compatibility.

## DIANA

The code implements largest-diameter divisive selection, maximum-mean initial splinter selection, positive maximum transfer contrasts, and recomputation after each transfer. Splitting stops before emptying either child. The tree conversion reverses the divisions in a topologically valid order. The coefficient uses each observation's final nonsingleton parent diameter divided by the original diameter.

Mean-absolute-deviation scaling is correct for stand=TRUE. Missing-coordinate Euclidean distances multiply squared available-coordinate sums by d/q before taking a square root; Manhattan distances multiply their absolute sums by d/q, where q is the available coordinate count. Missing values are retained in the original votes input, rather than silently imputing or dropping states.

The saved four source comparisons cover USArrests, votes Manhattan/Euclidean and raw agriculture. Independently added three fresh R fixtures: 31-by-3 rounded Gaussian data from NumPy seeds 19,217,3001, with two missing cells, MAD scaling and Manhattan DIANA. Complete cophenetic matrices match fresh cluster::diana within 1.78e-15, and coefficient differences are at most 1.11e-16. These fixtures check different inputs and missing patterns, not merely the published source outputs. Leaf orientation is not claimed identical. Ties use the disclosed smallest-row rule; arbitrary tied or degenerate inputs were not exhaustively compared.

## clustlearn, not DIANA

The installed primary implementation confirms depth-first left-before-right binary kmeans divisions, the requested five-iteration cap, first-minimum assignments, retention of empty centers, final reassignment, and the hierarchy's empty-child safeguard. Heights are sqrt(parent total SSE); they are not DIANA diameters or Ward increments. Merge sorting and leaf-order construction reproduce the package's layout. The helper's Python RNG is independent of R; optional replayed center arrays provide a controlled algorithm comparison, not recovery of an unknown source session.

Independently replayed all six fresh-R initial-center sequences on all six original 500-by-2 package tables. Every 499-row merge matrix and full leaf order agrees exactly; maximum absolute height error is 4.80e-14. Duplicate-row handling is an explicit safeguard. The fixed-start comparison is strong evidence for the implemented mechanics; it does not establish identity of independently drawn stochastic hierarchies.

## Hartigan–Wong and R-compatible sampling

Inspected initialization, nearest and second-nearest assignments, live-cluster optimal transfers, size-corrected add/remove costs, singleton protection, NCP/ITRAN updates, quick-transfer stopping, iteration faults and final WSS recomputation against R AS136. The wrapper preserves first-occurrence distinct input rows, numeric versus explicit initial centers, restart draws, strict improvement selection and continued RNG objects. It implements the relevant default MT19937/Rejection stream rather than delegating to NumPy or sklearn random-state defaults. Weighted sampling and other RNGkind/sample.kind modes are explicitly excluded.

Saved evidence compares six fits, 2,000 uniforms, 53 sequential sample entries, source WSS/silhouette curves and standardized-wine NbClust. Independently executed three fresh R fits on independent 31-by-3 rounded Gaussian inputs with seeds 19,217,3001, k=4,nstart=3: partitions agree exactly modulo labels and WSS agrees to floating precision. The seed-19 fixture exposes a real numerical boundary: equivalent restarts can have totals differing by a few rounding units, causing the strictly selected winning restart and label order to differ. Its single-start run and draw sequences agree. This is not a course-reference failure or a distinct geometric partition.

Resolved finding: added a bounded API docstring explaining that near-tied restart totals may change the winning restart/label numbering across floating-point summation implementations. No numerical behavior changed. Current helper SHA-256 after this documentation repair: `09ab1c311d205d211d45d76047b2b4c8a5b024e3a6e754e7af816b2d8792c7c1`. The comparison record must carry this current hash when claiming the current helper was assessed.

## Main laboratory and editorial implications

The main labs retain separate geometries for Utilities hierarchy versus kmeans, raw versus standardized wine, and MAD DIANA versus sample-SD kmeans. They distinguish controlled replacement seeds from recovered historical streams, single-start plotting wrappers from custom multi-start loops, Hartigan selection from the Hartigan–Wong optimizer, inverse-power jump from raw error decreases, and partition membership from projected convex hulls. The raw-wine threshold first reaches <=10 at H(16), shown under next-count 17; it does not substantiate a three-cluster threshold choice. USArrests' final four-cluster fit and two-cluster silhouette preference are correctly separate claims.

Remaining limitations are numerical and evidentiary boundaries, not untranslated course algorithms: unavailable original random session states, source-version sensitivity, stochastic null curves with different streams, near-tied restart ordering, and finite fixture coverage. This review does not certify all future inputs, legal licensing sufficiency, browser rendering, source-free packaging or any publication state.
