import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from models.lipschitz import lipschitz_regressor
from AL.exp_err_red import exp_upper, grad_exp_upper
from AL.tolerance.GP import get_multistarts
from AL.L2_GP import deps_dW

def solve_acc_prob(candidates, W, lips : lipschitz_regressor, samples, cost ) :
    n_cands = len(candidates)

    training_p = lips.train_x
    n_train = len(training_p)
    
    current_accs = np.mean(lips.noise, axis = 1)

    target_pts = np.concatenate( (training_p, candidates), axis = 0)
    current_precs = current_accs**(-1/cost) 

    _, LB_p, UB_p = lips.predict(target_pts, return_bds=True)
    LB_samples, UB_samples, Ldist = lips.predict_acc(samples, oth_tr=candidates)

    starts = get_multistarts(n_cands, W, current_precs)
    current_precs = starts[0]

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
                       args = ( n_train, LB_p, UB_p, samples, LB_samples, UB_samples, Ldist, lips, cost), 
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
    best_accs = np.ones( (1,lips.dout)) * best_accs.reshape( (-1,1))
    
    return best_accs, best_candidates, updated[:n_tr_pts]

def acc_prob_target(precs, n_train, LB_p, UB_p, samples, LB_samples, UB_samples, Ldist, lips, cost):

    accs = precs[precs>0]**(-1/cost)

    accs_tr = accs[:n_train]
    accs_new = accs[n_train:]

    new_LB, new_UB = lips.predict_acc(samples, accs_tr, Ldist[:,:n_train,:])

    EU = UB_samples - new_UB
    EL = LB_samples - new_LB
    #print(EU.mean(), EL.mean())
    for i in range(len(accs_new)) :
        alpha = UB_samples - Ldist[:,n_train +i] - accs_new[[i]]

        EU_pt = exp_upper(alpha, accs_new[[i]], LB_p[[n_train +i]],  UB_p[[n_train +i]], componentwise=True)

        beta = -LB_samples - Ldist[:,n_train +i] - accs_new[[i]]
        EL_pt = - exp_upper(beta,accs_new[[i]],-UB_p[[n_train +i]], -LB_p[[n_train +i]], componentwise=True)
        #print(EU_pt.mean())

        # EU = np.max( np.array([EU, EU_pt]), axis = 0)
        # EL = np.min( np.array([EL, EL_pt]), axis = 0)
        EU = EU + EU_pt
        EL = EL + EL_pt


    
    return - np.mean(np.sum( EU - EL, axis = 1), axis = 0 )



def acc_prob_grad(precs, n_train, LB_p, UB_p, samples, LB_samples, UB_samples, Ldist, lips, cost):
    accs = precs[precs>0]**(-1/cost)

    accs_tr = accs[:n_train]
    accs_new = accs[n_train:]

    new_LB, new_UB, dLB_deps, dUB_deps = lips.predict_acc(samples, accs_tr, Ldist[:,:n_train,:], return_deps = True)

    # EU = UB_samples - new_UB
    # EL = LB_samples - new_LB

    dEU_daccs = np.concatenate((-dUB_deps, np.zeros((len(accs_new), len(samples), lips.dout))), axis = 0)
    dEL_daccs = np.concatenate((-dLB_deps, np.zeros((len(accs_new), len(samples), lips.dout))), axis = 0)

    for i in range(len(accs_new)) :
        alpha = UB_samples - Ldist[:,n_train +i] - accs_new[[i]]
        dalpha_dacc = -1

        # EU_pt = exp_upper(alpha, accs_new[[i]], LB_p[[n_train +i]],  UB_p[[n_train +i]], componentwise=True)

        dEU_dalpha, dEU_deps, _, _ = grad_exp_upper(alpha, accs_new[[i]], LB_p[[n_train +i]],  UB_p[[n_train +i]])

        grad = dEU_dalpha * dalpha_dacc + dEU_deps 

        # cond = np.array(EU <= EU_pt)
        # dEU_daccs[:,cond] = 0
        # dEU_daccs[n_train +i,cond] = grad[cond]
        dEU_daccs[n_train + i] = grad

        beta = -LB_samples - Ldist[:,n_train +i] - accs_new[i]
        dbeta_dacc = -1

        # EL_pt = - exp_upper(beta,accs_new[[i]],-UB_p[[n_train +i]], -LB_p[[n_train +i]], componentwise=True)

        dEL_dbeta, dEL_deps, _, _ = grad_exp_upper(beta, accs_new[[i]], -UB_p[[n_train +i]], -LB_p[[n_train +i]])
        dEL_dbeta = - dEL_dbeta

        grad = dEL_dbeta * dbeta_dacc + dEL_deps
        # print(grad[cond])

        # cond = np.array(EL >= EU_pt)
        # dEL_daccs[:,cond] = 0
        # dEL_daccs[n_train +i,cond] = grad[cond]
        dEL_daccs[n_train + i] = grad

        # EU = np.max( np.array([EU, EU_pt]), axis = 0)
        # EL = np.min( np.array([EL, EL_pt]), axis = 0)

    dEI_daccs = np.mean(np.sum( dEU_daccs - dEL_daccs, axis = 2), axis = 1 )

    dEI_dprecs = np.zeros(len(precs))
    # print(dEI_daccs)

    dEI_dprecs[precs>0] = dEI_daccs * deps_dW(accs, cost)
    
    return - dEI_dprecs

