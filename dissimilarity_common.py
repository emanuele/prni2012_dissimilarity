"""
Experiments of the paper 'The Approximation of the Dissimilarity
Projection' accepted at PRNI2012.

Functions to select prototypes, to compute correlations between
distances in original space and projected space and to produce
paper figures.

Copyright (c) 2012, Emanuele Olivetti

Distributed under the New BSD license (3-clauses)
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr as correlation
from sys import stdout


def furthest_first_traversal(S, k, distance, permutation=True):
    """This is the farthest first traversal (fft) algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See for example:
    Hochbaum, Dorit S. and Shmoys, David B., A Best Possible Heuristic
    for the k-Center Problem, Mathematics of Operations Research, 1985.

    or: http://en.wikipedia.org/wiki/Metric_k-center
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    if permutation:
        idx = np.random.permutation(S.shape[0])
        S = S[idx]       
    else:
        idx = np.arange(S.shape[0], dtype=np.int)
    T = [0]
    while len(T) < k:
        z = distance(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]


def subset_furthest_first(S, k, distance, permutation=True, c=2.0):
    """Stochastic scalable version of the fft algorithm based in a
    random subset of a specific size.

    See: D. Turnbull and C. Elkan, Fast Recognition of Musical Genres
    Using RBF Networks, IEEE Trans Knowl Data Eng, vol. 2005, no. 4,
    pp. 580-584, 17.
    """
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        idx = np.random.permutation(S.shape[0])[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, distance, permutation=False)]


def compute_correlation(data, distance, prototype_policies, num_prototypes, iterations, verbose=False, size_limit=1000):
    print "Computing distance matrix and similarity matrix (original space):",
    data_original = data
    if data.shape[0] > size_limit:
        print
        print "Datset too big: subsampling to %s entries only!" % size_limit
        data = data[np.random.permutation(data.shape[0])[:size_limit], :]
    od = distance(data, data)     
    print od.shape
    original_distances = squareform(od)

    rho = np.zeros((len(prototype_policies), len(num_prototypes),iterations))

    for m, prototype_policy in enumerate(prototype_policies):
        print prototype_policy
        for j, num_proto in enumerate(num_prototypes):
            print "number of prototypes:", num_proto, " - ", 
            for k in range(iterations):
                print k,
                stdout.flush()
                if verbose: print("Generating %s prototypes as" % num_proto),
                # Note that we use the original dataset here, not the subsampled one!
                if prototype_policy=='random':
                    if verbose: print("random subset of the initial data.")
                    prototype_idx = np.random.permutation(data_original.shape[0])[:num_proto]
                    prototype = [data_original[i] for i in prototype_idx]
                elif prototype_policy=='fft':
                    prototype_idx = furthest_first_traversal(data_original, num_proto, distance)
                    prototype = [data_original[i] for i in prototype_idx]
                elif prototype_policy=='sff':
                    prototype_idx = subset_furthest_first(data_original, num_proto, distance)
                    prototype = [data_original[i] for i in prototype_idx]                
                else:
                    raise Exception                

                if verbose: print("Computing dissimilarity matrix.")
                data_dissimilarity = distance(data, prototype)

                if verbose: print("Computing distance matrix (dissimilarity space).")
                dissimilarity_distances = pdist(data_dissimilarity, metric='euclidean')

                rho[m,j,k] = correlation(original_distances, dissimilarity_distances)[0]
            print
    return rho


def plot_results(rho, num_prototypes, prototype_policies, color_policies):
    plt.figure()
    
    for m, prototype_policy in enumerate(prototype_policies):
        mean = rho[m,:,:].mean(1)
        std = rho[m,:,:].std(1)
        errorbar = std # 3.0 * std / np.sqrt(rho.shape[2])
        
        plt.plot(num_prototypes, mean, color_policies[m], label=prototype_policies[m], markersize=8.0)
            
        plt.fill(np.concatenate([num_prototypes, num_prototypes[::-1]]),
                 np.concatenate([mean - errorbar, (mean + errorbar)[::-1]]),
                 alpha=.25,fc='black',  ec='None')
    plt.legend(loc='lower right')
    plt.xlabel("number of prototypes $(p)$")
    plt.ylabel("correlation $\\rho(d, \Delta_{\Pi}^d)$")
    plt.show()
    
    
