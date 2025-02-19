import numpy as np

from scipy.optimize import minimize, Bounds

from AL.exp_err_red import exp_upper, grad_exp_upper

from utils.utils import latin_hypercube_sampling as lhs

def solve_pos_prob(n_pts, dom, lips, default_tol, samples) :
    mins = dom['min']
    maxs = dom['max']
    dim = len(maxs)
    
    # number of starts for multistarts
    n_starts = 10 * n_pts
    
    # evenly spread on parameter space
    starts = lhs(mins, maxs, n_starts)

    _, LB_samples, UB_samples = lips.predict(samples, return_bds=True)

    domain = Bounds(lb = mins, ub = maxs, keep_feasible = True )
    
    maxs = np.zeros_like(starts)
    vals = np.zeros(len(starts))
    
    iter_options = { 'maxiter' : 200,
                    'ftol' : 1.0e-8,
                    }
    
    # solve for each start
    for i, start in enumerate(starts) :
        
        res = minimize( pos_prob_target, start, 
                                        args = ( default_tol, lips, samples, LB_samples, UB_samples ),
                                        method='SLSQP',
                                        jac = pos_prob_grad,
                                        bounds=domain,
                                        options=iter_options
                                        )
        maxs[i] = res.x
        vals[i] = - res.fun
        print(res)

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
        
        close = np.isclose( (maxs- maxs[l])**2, 0, rtol=0, atol=0.03**2).all(axis = 1)
        
        maxs = maxs[~close]
        vals = vals[~close]
    
    return pts

def pos_prob_target(p, eps, lips, samples, LB_samples, UB_samples,):

    _, LB_p, UB_p = lips.predict(p, return_bds=True)

    dist = np.linalg.norm(p-samples, ord = 2, axis=1, keepdims=True)

    L = lips.L

    alpha = UB_samples - L * dist - eps

    EUI = exp_upper(alpha, eps, LB_p,  UB_p)

    beta = -LB_samples - L * dist - eps

    ELI = - exp_upper(beta,eps,-UB_p, -LB_p)


    return - np.mean(EUI - ELI )

def pos_prob_grad(p, eps, lips, samples, LB_samples, UB_samples,):
    p = p.reshape( (1,-1))

    _, LB_p, UB_p, dLB_dp, dUB_dp = lips.predict(p, return_bds=True, return_dp=True)

    dist = np.linalg.norm(p-samples, ord = 2, axis=1, keepdims=True)

    L = lips.L

    alpha = UB_samples - L * dist - eps
    beta = - LB_samples - L * dist - eps

    ddist_dp = -1/dist * (p-samples)
    dalpha_dp = L * np.transpose(ddist_dp).reshape((len(samples[0]),len(samples), -1) )
    dbeta_dp = dalpha_dp

    dEU_dalpha, _, dEU_dLB, dEU_dUB = grad_exp_upper(alpha, eps, LB_p, UB_p)

    EU_derivative = dalpha_dp*dEU_dalpha + dLB_dp * dEU_dLB + dUB_dp * dEU_dUB

    dEL_dbeta, _, dEL_dUB, dEL_dLB = grad_exp_upper(beta, eps, -UB_p, -LB_p)
    dEL_dbeta = - dEL_dbeta

    EL_derivative = dbeta_dp*dEL_dbeta - dUB_dp * dEL_dLB - dUB_dp * dEL_dUB

    derivative = EU_derivative - EL_derivative

    return - np.mean(np.sum(derivative , axis = 2 ), axis = 1)
