##########################################################################################
#
# Synthetic simulation, to test the constructed algorithm approach against various scenarios
# Uncomment to work with synthethic data analysis using - https://pyod.readthedocs.io/en/latest/
#
##########################################################################################

from sklearn.metrics.pairwise import euclidean_distances
from rbda import RBOD
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from pyod.utils import generate_data

from pyod.utils.data import generate_data
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.utils.data import get_outliers_inliers
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score
import pyod
from pyod.utils.example import data_visualize
from pyod.utils.example import visualize
import pandas as pd

contamination = 0.1
X_train, X_test, y_train, y_test = generate_data_clusters(n_train=100, n_test=50,
                            n_clusters=4, n_features=15,
                            contamination=contamination, size="different", density="different",
                            random_state=11, return_in_clusters=False)

##
# 4 clusters
# 30 f, 15 outliers, clusters have same size and different density.
# random_state=11 --> Constant

from sklearn.metrics import roc_auc_score
# Create the similarity matrix
C = np.zeros((X_train.shape[0], X_train.shape[0]))
# A simple euclidean distance over the synthethic dataset. Not against our similarity
for i in range(0, len(X_train)):
    for j in range(0, len(X_train)):
        dist = np.linalg.norm(X_train[i].reshape(1, -1) - X_train[j].reshape(1, -1))
        C[i][j] = dist

C_df = pd.DataFrame(C)
id = np.asarray([i for i in range(len(X_train))]).reshape(len(X_train), 1)
C_df.insert(0, "id", id)
X_train = np.hstack((id, X_train))
#clf = KNN(n_neighbors=k)
for k in range(10, 61, 1):
    combination_dict = {}
    rbod = RBOD(C_df, kneighbors=k)
    combination_dict["outliers"] = rbod.detect(X_train)

    # This code based on numpy executions of precision_scoring
    rbod_decision_scores = np.asarray([val[1] for val in combination_dict["outliers"]])
    #threshold = np.percentile(rbod_decision_scores, 100 * (1 - contamination))
    threshold = 2.5
    rbod_labels = (rbod_decision_scores > threshold).astype('int')
    #print("Classifier RBDA Outlier labels are - {}".format(rbod_labels))

    from pyod.utils import evaluate_print

    roc_rbod = np.round(roc_auc_score(y_train,
                                      [val[1] for val in combination_dict["outliers"]]), decimals=4)
    print("AUC Score - {} for k - {} ".format(roc_rbod, k))
