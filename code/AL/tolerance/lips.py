import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from models.lipschitz import lipschitz_regressor
from AL.tolerance.GP import get_multistarts
from AL.L2_GP import deps_dW

def solve_acc_prob(candidates, W, lips : lipschitz_regressor, samples, cost ) :
    n_cands = len(candidates)

    training_p = lips.train_x
    
    current_accs = np.mean(lips.noise, axis = 1)

    target_pts = np.concatenate( (training_p, candidates), axis = 0)
    current_precs = current_accs**(-cost) 

    mean_p = lips.predict(target_pts)
    Ldist = lips.compute_Ldist(samples, oth_tr=candidates)

    n_cands = len(candidates)    
    current_precs = np.concatenate( (current_precs, np.zeros( n_cands) ) )
    op_grad = acc_prob_grad(current_precs, mean_p, Ldist, lips, cost)

    starts = get_multistarts(n_cands, W, current_precs, op_grad)

    # budget constraint, spend all of the budget
    budget_constraint = LinearConstraint( np.ones_like(current_precs), lb = np.sum(current_precs) + W, ub = np.sum(current_precs) + W)
    # old points and positivity constraint, cannot decrease acccuracy in the old points
    old_points_constraint = Bounds(lb = current_precs, keep_feasible = True )

    iter_options = { 'maxiter' : 200,
                    'disp' : False, 
                    'ftol' : 1.0e-8,
                    }
    
    res_list = []
    # iterate over starts
    for j, start in enumerate(starts) :
        # scipy solve
        res = minimize( acc_prob_target, start, 
                       args = ( mean_p, Ldist, lips, cost), 
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
    # print(f'Best work distribution: {best_res}')

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

def acc_prob_target(precs, mean_p, Ldist, lips, cost):

    accs = precs[precs>0]**(-1/cost)
    mean_p = mean_p[precs>0]
    Ldist = Ldist[:,precs>0,:]

    new_LB, new_UB = lips.predict_acc(Ldist, mean_p, accs)
    
    return np.mean(np.sum(new_UB - new_LB, axis = 1), axis = 0 )



def acc_prob_grad(precs, mean_p, Ldist, lips, cost):

    accs = precs[precs>0]**(-1/cost)
    mean_p = mean_p[precs>0]
    Ldist = Ldist[:,precs>0,:]

    _, _, dLB_daccs, dUB_daccs = lips.predict_acc(Ldist, mean_p, accs, return_deps = True)

    dE_daccs = np.mean(np.sum( dUB_daccs - dLB_daccs, axis = 2), axis = 1 )

    dE_dprecs = np.zeros(len(precs))
    # print(dEI_daccs)

    dE_dprecs[precs>0] = dE_daccs * deps_dW(accs, cost)
    
    return dE_dprecs

