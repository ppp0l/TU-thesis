import numpy as np
import torch

from copy import deepcopy
import itertools as it

from scipy.optimize import minimize, Bounds, LinearConstraint

from AL.L2_GP import L2_approx, dL2_dk, deps_dW

def solve_acc_prob(candidates, W, GP, samples, std_samples, cost, talk = False) :
    """
    Solves the accuracy problem, i.e. how to distribute the computational budget among the points.
    Uses scipy.optimize.minimize, SLSQP method, multistarts.
    Problem is solved for computational budget, so that constraints are liearized.

    Args:
        candidates (np.ndarray): candidate points.
        W (float): computational budget.
    
    Returns:
        accs (np.ndarray): new evaluation tolerances.
        candidates (np.ndarray): new points.
        updated (np.ndarray): which old points have to be updated up to a new tolerance.
    """
    ### probably wrong here
    training_p = GP.train_X
    precs = np.sqrt(GP.noise.detach().numpy())**(-cost)
    dim = len(training_p[0])
    dout = len(std_samples[0])

    # old points and candidates
    target_pts = torch.cat( (torch.tensor(training_p), torch.tensor(candidates) ), dim = 0)
    
    # get samples
    samples = torch.tensor(samples).reshape( (-1, dim))
    
    # kernel matrices are needed multiple times
    ker_cand_samples = GP.kernel(target_pts, samples ).to_dense().detach().numpy()
    ker_cand = GP.kernel(target_pts, target_pts ).to_dense().detach().numpy()
    
    # recover scale factor in the kernel
    raw_var = GP.model.covar_module.task_covar_module.raw_var 
    constraint = GP.model.covar_module.task_covar_module.raw_var_constraint
    scale= constraint.transform(raw_var).detach().numpy()

    # helps avoiding weird numbers
    norm = L2_approx(std_samples**2).mean()

    n_cands = len(candidates)
    
    old_precs = np.concatenate( (precs, np.zeros( n_cands) ) )
    op_grad = acc_prob_grad(old_precs, dout, cost, ker_cand_samples, ker_cand, scale, norm )
    
    # multistarts
    starts = get_multistarts( n_cands, W, old_precs, op_grad)
    

    # budget constraint, spend all of the budget
    budget_constraint = LinearConstraint( np.ones_like(old_precs), lb = np.sum(old_precs) + W, ub = np.sum(old_precs) + W)
    # old points and positivity constraint, cannot decrease acccuracy in the old points
    old_points_constraint = Bounds(lb = old_precs, keep_feasible = True )
    
    iter_options = { 'maxiter' : 200,
                    'disp' : talk, 
                    'ftol' : 1.0e-8
                    }
    
    res_list = []
    

    # iterate over starts
    for j, start in enumerate(starts) :
        # scipy solve
        res = minimize( acc_prob_target, start, args = (dout, cost, ker_cand_samples, ker_cand, scale, norm ), 
                                        method='SLSQP',
                                        jac = acc_prob_grad,
                                        constraints = (budget_constraint), 
                                        bounds = old_points_constraint,
                                        options = iter_options
                                        )
        
        res_list.append(res)

    # dictionaries with optimization data  
    # acc_sol_pars=res_list
    
    # recover best solution
    best =np.inf
    best_precs = old_precs
    for res in res_list :
        if res.fun < best :
            best = res['fun']
            best_precs = res['x']
            best_res = res
    print(f'Best work distribution: {best_res}')

    # updated points, with threshold
    updated = (best_precs - old_precs) > 0.02*W
    n_tr_pts = len(training_p)
    print(f'Number of updated points: {np.sum(updated[:n_tr_pts])}' )
    print(f'Number of included candidates: {np.sum(updated[n_tr_pts:])}' )
    # non updated points
    best_precs[~updated] = old_precs[~updated]
    # spread evenly work from non updated
    best_precs[updated] += (W - np.sum(best_precs - old_precs))/(len(best_precs[updated]))
    
    # get best candidates, discards not included ones
    best_candidates = candidates[best_precs[n_tr_pts:]>0]
    # recovers tolerance
    best_accs = best_precs[best_precs > 0]**(-1/cost)
    
    return best_accs, best_candidates, updated[:n_tr_pts]


def get_multistarts(n_cands, W, precs, grad) :
    """
    Multistarts for the accuracy problem.
    """
    
    current = precs
    
    starts = [current]
    
    n_old = len(precs) - n_cands

    percentage = n_old//4
    threshold = np.partition( np.abs(grad), -percentage)[-percentage]

    best_indices = np.array(np.abs(grad) >= threshold)

    p = 0.5
    for index in best_indices :
        ns = deepcopy(current)
        ns[index] += p*W
        starts.append(ns)
    
    # add some random starts
    n_rand = n_old//4
    p = 0.7
    random_indices = np.array([np.full(len(current),False) for _ in range(n_rand)])
    rand_int = np.random.randint(0, len(current), size = (n_rand, n_old // 5 + 1))
    for i, ints in enumerate(rand_int) :
        random_indices[i][ints] = True

    for indeces in random_indices :
        ns = deepcopy(current)
        ns[indeces] += p*W/np.sum(indeces)
        starts.append(ns)
                      
    qs = [0.2, 0.4, 0.6]

    for vec in it.product([0,1], repeat = n_cands) :
        
        vec = np.array(vec, dtype = int)
        
        n_in = vec.sum()
        
        if n_in == 0 : continue 

        for q in qs: 
            ns = deepcopy(current)
            ns[ n_old : ] += q*W/n_in * vec
    
            starts.append(ns)

    return starts

def acc_prob_target( precs, dout, cost, ker_cand_samples, ker_cand, scale, norm) :
    """
    Target function for the accuracy problem.
    """    
    accs = precs[precs>0]**(-1/cost)
    
    k_samples = ker_cand_samples[ precs > 0 ]
    k_cand = ker_cand[precs > 0][:, precs > 0]
    
    k = compute_var( accs, dout, k_samples, k_cand, scale, return_daccs = False)
    
    L = L2_approx(k).mean()
    
    return  L/norm

def acc_prob_grad(precs, dout, cost, ker_cand_samples, ker_cand, scale, norm) :
    """
    Gradient of the target function for the accuracy problem.
    """
    dL_dprecs = np.zeros(len(precs))
    accs = precs[precs>0]**(-1/cost)
    
    k_samples = ker_cand_samples[ precs > 0 ]
    k_cand = ker_cand[precs > 0][:, precs > 0]
    
    k, dk_daccs = compute_var( accs, dout, k_samples, k_cand, scale, return_daccs = True)
    
    dl_dk = dL2_dk(k)
    
    dL_daccs = np.mean(np.sum(dk_daccs * dl_dk, axis = 2), axis = 1 )
    
    dL_dprecs [precs>0] = dL_daccs * deps_dW(accs, cost)
    return dL_dprecs / norm

def compute_var(accs, dout, ker_o, ker_oo, scale,
                    return_daccs = False) :
    """
    Computes the predictive variance of the GP as a function of the tolerances.
    Can also return the gradient of the variance with respect to the tolerances.
    """
    
    est = np.zeros( (len(ker_o[0]), dout) )
    
    if return_daccs :
            
        dk_daccs = np.zeros( (len(accs), len(ker_o[0]), dout) )
    
    for i, sc in enumerate(scale) :
        
        sk_o = sc * ker_o
        sk_oo = sc * ker_oo
        sk = sk_oo[0,0]
        
        sk_oo += np.diag(accs**2)
        
        prod = np.linalg.solve(sk_oo, sk_o)
        
        est[:,i] = sk - (sk_o * prod ).sum(axis = 0)
        
        if return_daccs :
                
            dk_daccs[:,:,i] = 2 * (prod.T**2 * accs).T
            
    if return_daccs :
        
        return est, dk_daccs
        
    return est