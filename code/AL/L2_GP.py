import numpy as np
import torch

from models.GP_models.MTSurrogate import MTModel

def L2_approx( k) :
    """
    L2 error model.
    """
    
    return k.sum(axis = 1)
 

def neg_dL2_dW( candidates : np.ndarray, dim : int, GP : MTModel, samples, cost, norm = 1) :
    
    candidates = np.reshape(candidates, (-1, dim))
    
    _, std = GP.predict( candidates, return_std = True)
    
    dL_dW = np.empty(len(candidates))
    
    dl_dk = dL2_dk(np.ones( (len(samples), len(std[0]))))
    
    for i, theta_i in enumerate(candidates) :
        
        dk_de = dk_deps(samples, torch.tensor(np.array([theta_i])), std[i] , GP)
        
        dL_dW[i] = np.mean(dl_dk * dk_de, axis = 0) @ (- deps_dW(std[i], cost))
    
    return - dL_dW / norm 


def dL2_dk(k) :
        
    return np.ones_like(k)
    

def dk_deps( pts, theta, std, GP : MTModel,) :
    # returns derivative of k w.r.t. to std
    # shape is len(pts) x dim_out

    training_p = GP.train_X
    errors = GP.noise
    dout = len(GP.train_y[0])
    
    if len(theta.shape) < 2 :
        theta = torch.tensor( [theta])
    
    if pts is not torch.Tensor :
        pts = torch.tensor(pts)
        
    training_p= torch.cat(( training_p, theta), dim = 0)

    
    k_star = GP.kernel( training_p, pts ).to_dense().detach().numpy()

    K = GP.kernel( training_p,  training_p ).to_dense().detach().numpy()
    
    # recover scale factor in the kernel
    raw_var = GP.model.covar_module.task_covar_module.raw_var 
    constraint = GP.model.covar_module.task_covar_module.raw_var_constraint
    scale= constraint.transform(raw_var).detach().numpy()
    
    dfull = np.zeros( (len(training_p), len(pts), dout) )

    for i, sc in enumerate(scale) :
        
        k_o = sc * k_star
        sK = sc * K
        
        sK += np.diag(np.append(errors**2, std.mean()**2))
        
        prod = np.linalg.solve(sK, k_o)
        
        dfull[:,:,i] = prod**2
    
    dfull = dfull[-1]
    
    return std*dfull


def deps_dW( accs, cost) :
    
    return -(1/cost) * accs**(cost+1)
