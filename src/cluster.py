"""
Code for clustering.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster
import fastcluster


class Clusterer:
    """
    Class for clustering. Here we try different types of clustering depending on the input parameters.
    """

    def __init__(
        self,
        X=None,
        distances=None,
        cluster_method=None,
        linkage_method=None
    ):
        """
        X is either a 1D condensed distance matrix or a 2D array of observation vectors.
        """
        if type(X) != type(None):
            assert X.dtype == "float32"
            self.X = X
        else:
            self.X = None
        self.distances = distances
        self.cluster_method = cluster_method
        self.linkage_method = linkage_method
        self.Z = None
        self.flat_clusters = None
        self.nn_distances = None
        self.nn_indices = None
        self.finchC = None

    def setZ(
        self,
        inputZ
    ):
        self.Z = inputZ

    def cluster(self):
        if self.cluster_method == 'hac':
            if self.distances is not None:
                self.Z = fastcluster.linkage(self.distances, self.linkage_method)
            else:
                self.Z = fastcluster.linkage(self.X, self.linkage_method)
        elif self.cluster_method == 'finch':
            self.finchC, num_clust, rec_q = FINCH(
                self.X, initial_rank=None, req_clust=None, distance='cosine', verbose=True)

    def compute_flat_clusters(
        self,
        distance=None,
        max_num_clusters=None
    ):
        """
        Computes the flat clusters.
        """
        if distance is not None:
            self.flat_clusters = fcluster(
                self.Z, t=distance, criterion="distance")
        elif max_num_clusters is not None:
            self.flat_clusters = fcluster(
                self.Z, t=max_num_clusters, criterion="maxclust")

    def extract_clusters(self):
        """
        Returns list L = [[idxa, idxb, ..], [], ... ]
        len(L) = number of clusters
        L[i] where i is the cluster index
        L[i][j] where j is the index in the original self.X features
        """
        assert self.cluster_ready()
        L = []
        for i in range(self.flat_clusters.min(), self.flat_clusters.max() + 1):
            indices = list(np.argwhere(self.flat_clusters == i).flatten())
            L.append(indices)
        return L

    def compute_nn(self):
        """
        Computes the nearest neighbors.
        """
        # TODO: switch this to faiss nn
        nbrs = NearestNeighbors(
            n_neighbors=10, algorithm='ball_tree').fit(self.X)
        self.nn_distances, self.nn_indices = nbrs.kneighbors(self.X)

    def cluster_ready(self):
        return (self.Z is not None and self.flat_clusters is not None)

    def nn_ready(self):
        return (self.nn_distances is not None and self.nn_indices is not None)

    def clear_computations(self):
        """
        Clear memory that isn't needed anymore.
        """
        self.flat_clusters = None
        self.nn_distance = None
        self.nn_indices = None
