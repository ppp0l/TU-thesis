import numpy as np

from scipy.optimize import minimize, Bounds

from AL import exp_err_red

from utils.utils import latin_hypercube_sampling as lhs

def pos_prob(n_pts, dom, lips, default_tol, samples) :
    mins = dom['min']
    maxs = dom['max']
    dim = len(maxs)
    
    # number of starts for multistarts
    n_starts = 10 * n_pts
    
    # evenly spread on parameter space
    starts = lhs(mins, maxs, n_starts)

    
    # find maxima
    domain = Bounds(lb = mins, ub = maxs, keep_feasible = True )
    
    maxs = np.zeros_like(starts)
    vals = np.zeros(len(starts))
    
    # solve for each start

    # sparsify
    pass