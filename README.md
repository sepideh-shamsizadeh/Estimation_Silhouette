# Estimation_Silhouette

Representation of points. We will work with points in Euclidean space (real cooordinates) and with the squared Euclidean L2-distance.

Represent points as the standard tuple of float (i.e., point = (x1, x2, ...)). Although Spark provides the class Vector also for Python (see pyspark.mllib package), its performance is very poor and its more convenient to use tuples, especially for points from low-dimensional spaces.

Writing a program, which receives in input, as command-line arguments, the following data (in this ordering)

-A path to a text file containing point set in Euclidean space partitioned into k clusters. Each line of the file contains, separated by commas, the coordinates of a point and the ID of the cluster (in [0,k-1]) to which the point belongs. E.g., Line 1.3,-2.7,3 represents the point (1.3,-2.7) belonging to Cluster 3. Your program should make no assuptions on the number of dimensions!

-The number of clusters k (an integer).

-The expected sample size per cluster t (an integer).

-The program must do the following:

1-Read the input data. In particular, the clustering must be read into an RDD of pairs (point,ClusterID) called fullClustering which must be cached and partitioned into a reasonable number of partitions, e.g., 4-8. 

2-Compute the size of each cluster and then save the k sizes into an array or list represented by a Broadcast variable named sharedClusterSizes. (Hint: to this purpose it is very convenient to use the RDD method countByValue() whose description is found in the Spark Programming Guide)

3-Extract a sample of the input clustering, where from each cluster C, each point is selected independently with probability min{t/|C|, 1} (Poisson Sampling). Save the sample, whose expected size is at most t*k, into a local structure (e.g., ArrayList in java or list in Python) represented by a Broadcast variable named clusteringSample. (Hint: the sample can be extracted with a simple map operation on the RDD fullClustering, using the cluster sizes computed in Step 2).

4-Compute the approximate average silhouette coefficient of the input clustering and assign it to a variable approxSilhFull. (Hint: to do so, you can first transform the RDD fullClustering by mapping each element (point, clusterID) of fullClustering to the approximate silhouette coefficient of 'point' computed as explained here exploiting the sample, and taking the average of all individual approximate silhouette coefficients). 

5-Compute (sequentially) the exact silhouette coefficient of the clusteringSample and assign it to a variable exactSilhSample.

6-Print the following values: (a) value of approxSilhFull, (b) time to compute approxSilhFull (Step 4),  (c) value of exactSilhSample, (d) time to compute exactSilhSample (Step 5). Times must be in ms. Use the following output format
