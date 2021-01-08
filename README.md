# Rank Based Detection Algorithm
This is a python based implementation of RBDA

# Set up
Operating System: Windows

1. This repo contains a requirements.txt using which one can install all the libraries on the fly at once.
    * cd into the extracted folder and run **pip install -r requirements.txt** . With all things in place this should setup all required libraries correctly

Now the system is ready for running the code.


## Usage:
-- Initialiation of the Class RBOD:
```
from rbda import RBOD
rbod = RBOD(sim_df, metric=None, kneighbors=5, z_val=2.5)
sim_df --> Is the pairwise similarity matrix across data objects. This matrix also requires a column called "id" for the reverse ranks computation. The z_score value is retained same as the original paper 2.5.
metric --> None means will keep it to euclidean else the passed metric (cosine, mannhattan) can be used.
```

### Other useful references:
-- Test classes:
1. test_rbda_simulate.py
   This runs over a simulated data using ```pyod``` and AUC scores are produced
2. test_rbda_bcw.ipynb
   This runs over a dataset and explains it step by step so it can be extended to other datasets.

### Original Paper:
> Huang, H., Mehrotra, K., & Mohan, C. K. (2013). Rank-based outlier detection. Journal of Statistical Computation and Simulation, 83(3), 518-531.
> https://doi.org/10.1080/00949655.2011.621124

### Other useful papers:
> Goldstein, M., & Uchida, S. (2016). A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data. PloS one, 11(4), e0152173.

> Bhattacharya, G., Ghosh, K., & Chowdhury, A. S. (2015). Outlier detection using neighborhood rank difference. Pattern Recognition Letters, 60, 24-31.