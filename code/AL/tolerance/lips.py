import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from models.lipschitz import lipschitz_regressor
from AL import exp_err_red
from AL.tolerance.GP import get_multistarts

def solve_acc_prob(W, candidates, samples, lips : lipschitz_regressor, cost ) :
    n_cands = len(candidates)

    training_p = lips.train_x
    current_accs = np.mean(lips.noise, axis = 1)

    target_pts = np.concatenate( (training_p, candidates), axis = 0)
    current_accs = np.concatenate( (current_accs, np.zeros(n_cands) ))
    current_precs = current_accs**(-1/cost) 

    _, LB_p, UB_p = lips.predict(target_pts, return_bds=True)

    samples2 = np.sum(samples**2, axis = 1, keepdims=True)
    tr_x2 = np.sum(target_pts**2, axis = 1, keepdims=True) 
    samplesxT = samples.dot(target_pts.transpose())
    dist_x = np.sqrt( samples2 + tr_x2.transpose() - 2 *samplesxT )
    dshape = dist_x.shape
    Ldist = np.outer(dist_x.reshape(-1 ), lips.L).reshape( (dshape[0],dshape[1],lips.dout))
    
    # budget constraint, spend all of the budget
    budget_constraint = LinearConstraint( np.ones_like(), lb = np.sum(current_precs) + W, ub = np.sum(current_precs) + W)
    # old points and positivity constraint, cannot decrease acccuracy in the old points
    old_points_constraint = Bounds(lb = current_precs, keep_feasible = True )

    starts = get_multistarts(n_cands, W, current_precs)
    
    iter_options = { 'maxiter' : 200,
                    'ftol' : 1.0e-8
                    }
    
    res_list = []
    
    # iterate over starts
    for j, start in enumerate(starts) :
        # scipy solve
        res = minimize( acc_prob_target, start, args = (LB_p, UB_p, samples, Ldist, lips, cost), 
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
    best_precs = current_precs
    for res in res_list :
        if res.fun < best :
            best = res['fun']
            best_precs = res['x']
            best_res = res
    print(f'Best work distribution: {best_res}')

    # updated points, with threshold
    updated = (best_precs - current_precs) > 0.02*W
    n_tr_pts = len(training_p)
    print(f'Number of updated points: {np.sum(updated[:n_tr_pts])}' )
    print(f'Number of included candidates: {np.sum(updated[n_tr_pts:])}' )
    # non updated points
    best_precs[~updated] = current_precs[~updated]
    # spread evenly work from non updated
    best_precs[updated] += (W - np.sum(best_precs - current_precs))/(len(best_precs[updated]))
    
    # get best candidates, discards not included ones
    best_candidates = candidates[best_precs[n_tr_pts:]>0]
    # recovers tolerance
    best_accs = best_precs[best_precs > 0]**(-1/cost)
    
    return best_accs, best_candidates, updated[:n_tr_pts]

def acc_prob_target(precs, LB_p, UB_p, samples, Ldist, lips, cost):

    accs = precs[precs>0]**(-1/cost)



def acc_prob_grad(precs, LB_p, UB_p, samples, Ldist, lips, cost):
    pass
