# Chapter 6: K-means and choosing a partition

Full transcript 6qNLWCODvz0 read sequentially. Selected 720p source frames
1347/1400/1460/1490/1520/1600/1710/1770/1830/1950/2070/2165 inspected.

1. Define fixed-k Euclidean SSE objective, partitions, centroids; prove the
   arithmetic mean minimizes squared error. Scaling defines the geometry.
2. Prove assignment/update descent, finite-partition/tie assumptions and local
   minima; hand-work source four-observation example through WSS28 ->27 ->14.
   Distinguish Lloyd from source R Hartigan-Wong default and random starts.
3. Establish optimal WSS nonincreasing in k, explain erroneous plotted local
   WSS increase; elbow, normalized distortion, inverse-power jump p/2, Hartigan
   threshold heuristic and silhouette formula including singleton convention.
4. Convert Utilities all8 standardized features k6, printed memberships,
   centroids/SSE/sizes/centroid distances and profile graphics. Contrast ch5's
   two-feature geometry.
5. Convert rawwine cultivar-excluded k3 single/25starts seed278613, PCA
   displays/confusion table, useful::FitKMeans max20/Hartigan. Separately convert
   standardized rattlewine seed1234 NbClust k2..15 defaultall26 call and WSS.
6. Convert USArrests scale/sampleSD, distance heatmap, nstart25 k2..5 PCA hull
   displays, nstart10 WSS curve seed123, silhouette nstart25 k2..15, final
   seed123 k4 nstart25 and rawdata cluster means. Record any exact-path gap.
7. Six fully solved conceptual/derivation/computational exercises. Execute all
   examples on actual canonical data, retain stdout and compare fresh R.
