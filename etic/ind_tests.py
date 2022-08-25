"""Test statistics.

Author: Lang Liu
"""

from __future__ import absolute_import, division, print_function

import os
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np

from .ind_criterion import etic, hsic


##########################################################################
# HSIC
##########################################################################
    
class HSICTest(object):
    """A class for independence testing with HSIC.
    """
    
    def __init__(self):
        pass
    
    def compute_pval(self, xmat, ymat, nperms, parallel=False, ncores=1):
        size = xmat.shape[0]
        stat = hsic(xmat, ymat)
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed(repeat)
            ind = np.random.choice(size, size=size, replace=False)
            return hsic(xmat, ymat[np.ix_(ind, ind)])

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return pval


##########################################################################
# ETIC with Tensor Sinkhorn
##########################################################################

class ETICTest(object):
    """A class for independence testing with ETIC.
    
    It is only implemented for continuous random variables.
    """
    
    def __init__(self):
        pass
    
    def compute_pval(self, xmat, ymat, reg, max_iter=1000, low_rank=False,
                     nperms=200, parallel=False, ncores=1):
        size = xmat.shape[0]
        stat = etic(xmat, ymat, reg, max_iter, low_rank)
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed(repeat)
            ind = np.random.choice(size, size=size, replace=False)
            if low_rank:
                return etic(xmat, ymat[ind, :], reg, max_iter, low_rank=True)
            return etic(xmat, ymat[np.ix_(ind, ind)], reg, max_iter)

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return pval
