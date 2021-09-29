from pyspark import SparkContext, SparkConf
import sys
import os
import time
import numpy as np

sharedClusterSizes = []
ti = []


def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch) - 1))
    return (point, int(ch[-1]))  # returns (point, cluster_index)


def sampling(doc, T):
    """
    Sampling points by considering the ti, the number of sample for each cluster
    Input doc: A tuple of (point, cluster ID)
    Input T: For computing probability for each point
    Output: Selected doc
    """
    r = np.random.uniform(0.0, 1.0)
    prob = min(1, T / sharedClusterSizes[doc[1]])
    if r < prob and ti[doc[1]] > 0:
        ti[doc[1]] -= 1
        return doc


def distance(point, point1):
    d = 0
    for i in range(0, len(point)):
        d += (point[i] - point1[i]) ** 2
    return d


def silhouette(point, clusters, K, Ti):
    """
    Computing silhouette coefficients
    Input point: A tuple = (point, cluster ID)
    Input clusters: The list of samples that was broadcasted
    Input K: The number of clusters
    Input Ti: The number of samples for each clusters
    Output: sp for each point
    """
    ap = 0
    bc = K * [0]
    for value in clusters:
        d = distance(point[0], value[0])
        if point[1] == value[1]:
            ap += d
        else:
            bc[value[1]] += d
    ap /= Ti[point[1]]
    bc = [bc[i] / Ti[i] for i in range(0, K)]
    bc.pop(point[1])
    bp = min(bc)
    return (bp - ap) / max(ap, bp)


def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python G03HW2.py <K> <file_name>"
    print(sys.argv)

    # SPARK SETUP
    conf = SparkConf().setAppName('G03HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING
    # 1. Read number of clusters
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read the expected sample size per cluster t
    T = sys.argv[2]
    assert T.isdigit(), "T must be an integer"
    T = int(T)
    if T >= 500:
        p = 8
    else:
        p = 4

    # 3. Read input file
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    fullClustering = sc.textFile(data_path, minPartitions=4).cache()
    fullClustering.repartition(numPartitions=p)

    # Map fullClustring from string to a tuple =(point, cluster ID)
    fullClustering = fullClustering.map(strToTuple)

    # Count the number of points in each cluster
    dict_values = fullClustering.values().countByValue()
    for i in range(0, K):
        sharedClusterSizes.append(dict_values[i])
        ti.append(min(T, dict_values[i]))  # Select the number of each cluster for sampling
    Ti = ti.copy()

    # Sampling and remove None from the result list
    l = list(filter(None, fullClustering.map(lambda x: sampling(x, T)).collect()))

    # Broadcasting clusteringSample
    clusteringSample = sc.broadcast(l)

    # Release the space of list l
    l.clear()

    # Computing approxSilhFull
    fullClustering.repartition(numPartitions=p)
    num_c = fullClustering.count()
    start_time = time.time()
    approxSilhFull = fullClustering.map(lambda x: silhouette(x, clusteringSample.value, K, Ti)).sum() / num_c
    end_time = time.time()
    elapsed_time_approx = int((end_time - start_time) * 1000)
    print("Value of approxSilhFull = ", approxSilhFull)
    print("Time to compute approxSilhFull = ", elapsed_time_approx, "ms")

    # Computing exactSilhSample
    rdd = sc.parallelize(clusteringSample.value).cache().repartition(numPartitions=p)
    num = rdd.count()
    start_time = time.time()
    exactSilhSample = rdd.map(lambda x: silhouette(x, clusteringSample.value, K, Ti)).sum()/num
    end_time = time.time()
    elapsed_time_exact = int((end_time - start_time)*1000)
    print("Value of exactSilhSample = ", exactSilhSample)
    print("Time to compute exactSilhSample = ", elapsed_time_exact, "ms")


if __name__ == "__main__":
    main()
