"""
Experiments of the paper 'The Approximation of the Dissimilarity
Projection' accepted at PRNI2012.

Quantification of the dissimilarity approximation of simulated 2D data
across different prototype selection policies and number of prototypes.

Copyright (c) 2012, Emanuele Olivetti

Distributed under the New BSD license (3-clauses)
"""

import numpy as np
from scipy.spatial.distance import cdist
from dissimilarity_common import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':

    np.random.seed(0)

    num_prototypes =range(1,20)
    prototype_policies = ['random', 'fft', 'sff']
    color_policies = ['ko--', 'kx:', 'k^-']
    iterations = 1000
    verbose = True
    
    data = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1,0],[0,1]], size=50)

    rho = compute_correlation(data, cdist, prototype_policies, num_prototypes, iterations, verbose=False)
    plot_results(rho, num_prototypes, prototype_policies, color_policies)

    print
    print "Plotting the dataset and 3 prototypes."
    prototypes_idx = furthest_first_traversal(data, 3, cdist) # np.random.permutation(data.shape[0])[:3]
    prototypes = data[prototypes_idx,:]
    plt.figure()     
    plt.plot(data[:,0], data[:,1], 'ko')    
    plt.plot(prototypes[:,0], prototypes[:,1], 'rx', markersize=16, markeredgewidth=4)

    print "Plotting the dissimilarity projected dataset."
    data_dissimilarity = cdist(data, prototypes, metric='euclidean')
    prototypes_dissimilarity = cdist(prototypes, prototypes, metric='euclidean')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data_dissimilarity[:,0], data_dissimilarity[:,1], data_dissimilarity[:,2], marker='o', c='k')
    ax.scatter(prototypes_dissimilarity[:,0], prototypes_dissimilarity[:,1], prototypes_dissimilarity[:,2], marker='x', color='red', s=400, linewidths=4)    

    plt.show()    
