"""
Experiments of the paper 'The Approximation of the Dissimilarity
Projection' accepted at PRNI2012.

Quantification of the dissimilarity approximation of tractography data
across different prototype selection policies and number of prototypes.

Copyright (c) 2012, Emanuele Olivetti

Distributed under the New BSD license (3-clauses)
"""

import numpy as np
import nibabel as nib
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.dpy import Dpy
from dissimilarity_common import *

if __name__ == '__main__':

    np.random.seed(0)

    figure = 'small_dataset' # 'big_dataset' # 

    if figure=='small_dataset':
        filename = 'data/subj_05/101_32/DTI/tracks_dti_10K.dpy'
        prototype_policies = ['random', 'fft', 'sff']
        color_policies = ['ko--', 'kx:', 'k^-']
    elif figure=='big_dataset':
        filename = 'data/subj_05/101_32/DTI/tracks_dti_3M.dpy'
        prototype_policies = ['random', 'sff']
        color_policies = ['ko--', 'k^-']
    num_prototypes = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    iterations = 50
    
    print "Loading tracks."
    dpr = Dpy(filename, 'r')
    tracks = dpr.read_tracks()
    dpr.close()
    tracks = np.array(tracks, dtype=np.object)
    # tracks = tracks[:100]
    print "tracks:", tracks.size

    rho = compute_correlation(tracks, bundles_distances_mam, prototype_policies, num_prototypes, iterations)
    plot_results(rho, num_prototypes, prototype_policies, color_policies)
