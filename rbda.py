"""
A voting based approach to detect outliers using the reverse neighborhood ranking and LOF is utilized.
Author: Subash Prakash
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class RBOD:
    """
    The notion of reverse neighborhood concept has been widely useful to detect varying densities point. There has been
    recent attention to this reverse neighborhood in many fields related to medical, social network etc..
    The reverse neighborhood means to ask if for a query q there are set of nearest neighbors, then how close are these
    neighors?. Closeness is defined by obtain the rank of the query in the neighbors. Ideally, for a neighbor p of q, q
    must be one of the neighbors. This is always not true and could lead to outliers. This class implements such a detection
    as per "Rank-based outlier detection" mostly a good start. Since, the paper do not have their implementation this is
    contributed in the thesis work.
    """

    def __init__(
        self, sim_df, kneighbors=5, metric=None, radius=1.0, z_val=2.5
    ):
        """
        :param sim_df: Precompute a similarity matrix by a defined metric and pass to the class.
        :param kneighbors: Number of K neighbors same as KNN
        :param metric: If None, default euclidean is perform, else the passed metric is used for computation same as sim_df
        :param radius: Defaults to the radius around query to 1.0, if there is a threshold in radius it can be passed.
        :param z_val: The cut-off for outlier as per the original paper is 2.5, however based on the domain this can change.
        """

        self.kneighbors = kneighbors
        self.metric = metric
        self.radius = radius
        self.sim_df = sim_df
        self.z_val = z_val

    def detect(self, X):
        """
        :param X: The processed Train data. This contains all the necessary information. This implementation also needs
        an id, this can be id, or just an index_id for the identifier between the sim_df and X_train.
        :param X: In case of already computed data, it would be a similarity matrix numpy processed array.
        :return: outlier_list contains a tuple (<<Id>>,<<z_val>>,<<True/False indicating the outlierness>>)
        """

        outlier_list = []
        if None is not self.metric:
            knn = NearestNeighbors(
                n_neighbors=self.kneighbors, radius=self.radius, metric=self.metric
            )
            # knn = NearestNeighbors(n_neighbors=20, radius=1.0, metric="precomputed")
        else:
            # This is the default Euclidean distance. Mostly useful while doing over public dataset
            knn = NearestNeighbors(n_neighbors=self.kneighbors)

        # Fit to ball_tree entire data.
        # We will obtain the neighbors indexed from this
        # matrix and use it find and label outliers if any.
        # First column is the id so ignoring it
        knn.fit(X[:, 1:])

        # Can be any index termed with colname as id
        ids_np = self.sim_df["id"].to_numpy()
        for obj in X:
            dist, idx = knn.kneighbors(
                obj[1:].reshape(1, -1), n_neighbors=self.kneighbors
            )

            # Not concerned with the distance but their ranks across the neighbors
            dist = dist.flatten()
            idx = idx.flatten()
            ranks_list = []
            for o_id in idx[1:]:
                obj_rank_dict = {}
                o_id_dists = self.sim_df[
                    self.sim_df["id"] == int(ids_np[o_id])
                ].to_numpy()
                for i, dists in enumerate(o_id_dists.flatten()[1:]):
                    if float(dists) != 0.0:
                        obj_rank_dict[ids_np[i]] = dists

                # Sort in ascending order and find the rank of the respective data_obj (represented by ids)
                sorted_usr_rank_tuple = sorted(
                    obj_rank_dict.items(), key=lambda x: x[1]
                )
                rank = 1
                for val in sorted_usr_rank_tuple:
                    if int(obj[0]) == val[0]:
                        break
                    rank += 1
                ranks_list.append(rank)
            outlier_list.append(
                (int(obj[0]), np.sum(ranks_list) / self.kneighbors))

        np_outlier_list = np.asarray([val[1] for val in outlier_list])
        mean_rank_ratio = np_outlier_list.mean()
        std_rank_ratio = np_outlier_list.std()
        z_list = []
        for ele in outlier_list:
            z_score = (ele[1] - mean_rank_ratio) / std_rank_ratio
            if z_score >= self.z_val:
                z_list.append((ele[0], z_score, True))
            else:
                z_list.append((ele[0], z_score, False))

        return z_list
