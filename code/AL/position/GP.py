import numpy as np
from scipy.optimize import minimize, Bounds

from utils.utils import latin_hypercube_sampling as lhs
from AL.L2_GP import L2_approx, neg_dL2_dW


def solve_pos_prob(n_pts : int, dom : dict, default_tol : float, GP, samples, std_samples : np.ndarray, cost : float, talk = False, talk_a_lot = False):
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
    
    norm = L2_approx(std_samples**2).mean()
    for i, start in enumerate(starts) :
        
        res = minimize( neg_dL2_dW, start, 
                                        args = ( dim, default_tol, GP, samples, cost, norm ),
                                        method='L-BFGS-B',
                                        bounds=domain,
                                        )
        if talk :
            print(res)
        maxs[i] = res.x
        vals[i] = - res.fun
    
    n_pts = min(n_pts, len(vals))
    
    pts = np.zeros((n_pts, dim))
    valori = np.zeros( n_pts)
    best = np.max(vals)
    
    # discard points that are too close to each other or with too small value
    for i in range(n_pts):
        if vals.size == 0 :
            pts = pts[:i]
            break
        
        l = np.argmax(vals)
        
        if vals[l] < best/20 :
            pts = pts[:i]
            break
        
        pts[i] = maxs[l]
        valori[i] = vals[l]
        
        close = np.isclose( (maxs- maxs[l])**2, 0, rtol=0, atol=0.02**2).all(axis = 1)
        
        maxs = maxs[~close]
        vals = vals[~close]
    
    return pts
