

import numpy as np
from matplotlib import pyplot as plt
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw
import random

random.seed(10)

# --- Toy data #1:
N = 20000 #Number of points
D = 2 # Dimension
modes = 20 # Number of Clusters
var_scale = 100.0 # The variance of the MV-Normal distribution where the clusters means are sampled from.

# --- Toy data #2:
#N = 10000  # number of points
#D = 2   # data dimension
#modes = 6   # number of modes
#var_scale = 80.0

# --- Extract the data:
data, labels = DPMMPython.generate_gaussian_data(N, D, modes, var_scale) 

# Changing the lables to be incorrect (to see how the splits work)
#labels[labels==3] = 2
#labels[labels==4] = 3
#labels[labels==5] = 4
#labels[labels==6] = 5

# --- hyper params #1:
hyper_prior = niw(1,np.zeros(D),5,np.eye(D)*0.5)
alpha = 10.
iters = 500

# --- hyper params #2:
#init_clusters = np.unique(labels).size
#m = np.zeros(D)
#k = init_clusters  #1.0
#nu = 130.  # should be > D
#psi = np.cov(data)*0.01  # shape (D,D)
#hyper_prior = niw(k, m, nu, psi)
#alpha = 1.
#iters = 200

# --- Print original label counts:
label_counts = np.zeros(init_clusters)
for i in range(len(labels)):
    l = int(labels[i]-1)
    label_counts[l] = label_counts[l] + 1

for i in range(len(label_counts)):
    print("label ", str(i+1), ": ", str(label_counts[i]))


# --- Run DP:
results = DPMMPython.fit(data, alpha, prior = hyper_prior, iterations=iters, outlier_params=labels, verbose=True)