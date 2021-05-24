

import numpy as np
from matplotlib import pyplot as plt
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw

D = 2 # Dimension
K = 20 # Number of Clusters
N = 20000 #Number of points
var_scale = 100.0 # The variance of the MV-Normal distribution where the clusters means are sampled from.
data, labels = DPMMPython.generate_gaussian_data(N, D, K, var_scale) 

prior = niw(1,np.zeros(D),5,np.eye(D)*0.5)
alpha = 10.0

results = DPMMPython.fit(data,alpha,prior = prior,iterations=500)