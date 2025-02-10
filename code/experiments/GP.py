"""
created on: 2025/01/19

@author: pvillani
"""
import argparse
from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.GP_models.MTSurrogate import MTModel

from IP.priors import FlatPrior
from IP.likelihoods import base_likelihood, GP_likelihood
from IP.posteriors import Posterior 

from AL.L2_GP import L2_approx
from AL.position.GP import solve_pos_prob
from AL.tolerance.GP import scipy_acc_prob

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed

parser = argparse.ArgumentParser()
parser.add_argument("--dim", default=2, type=int, help="Dimension of parameter space")
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
dim = args.dim
path = args.path

# manages i/o
workflow_manager = Manager(path, dim)

# loads configuration parameters
#configuration = workflow_manager.read_parameters()
value = np.array([0.36768606, 0.83379858])

# sets seed
reproducibility_seed(seed = 1)

### actual task

# parameter space
param_space = { 
    'min' : np.zeros(dim), 
    'max' : np.ones(dim)
}

# create forward model,  sets noise type
forward = fm(dim, "N", dom = param_space)

# create prior
prior = FlatPrior(param_space)

# create true likelihood
meas_cov = 0.01 * np.ones(forward.dout)
true_likelihood = base_likelihood(value, meas_cov, forward)

# create true posterior
true_posterior = Posterior(true_likelihood, prior)

# active learning parameters
n_init = 5
points_per_it = dim
n_it = 5


default_tol = 0.05
FE_cost = 1
budget = points_per_it * (default_tol)**(-FE_cost)

# create surrogate
surrogate = MTModel(num_tasks = forward.dout)
train_p = lhs(param_space["min"], param_space["max"], n_init)
train_y, errors = forward.predict(train_p, tols = default_tol * np.ones(n_init))
surrogate.fit(train_p, train_y, errors**2)
# export initial surrogate

# create approximate likelihood and posterior
approx_likelihood = GP_likelihood(value, meas_cov, surrogate)
approx_posterior = Posterior(approx_likelihood, prior)

n_walkers = 16
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
approx_posterior.initialize_sampler(n_walkers, initial_pos)

n_samples = 0
samples = np.array([np.zeros(dim)])
for type_run in ["fullyAd"] :
    # load initial
    for i in range(n_it) :
        n_burn = n_samples - 10
        n_samples += 100 
        # sample posterior
        new_samples = approx_posterior.sample_points(n_samples)

        # update sample chains
        samples = samples[n_burn :]
        samples = np.concatenate( (samples, new_samples), axis = 0)

        _, std = surrogate.predict(samples, return_std=True)

        curr_L2 = L2_approx(std).mean()

        # position problem
        candidates = solve_pos_prob(2, param_space, surrogate, samples, std, FE_cost )

        # accuracy problem
        tolerances, new_pts, updated = scipy_acc_prob(candidates, budget, surrogate, samples, std, FE_cost)

        new_tols = tolerances[-len(new_pts):]
        update_tols = tolerances[:-len(new_pts)]

        # evaluate model
        if len(new_pts) > 0:
            new_vals, new_errs = forward.predict(new_pts, new_tols)
        if np.any(updated) :
            train_y[updated], errors[updated] = forward.predict(train_p[updated], update_tols[updated])

        # update surrogate
        train_p =  np.concatenate((train_p, new_pts), axis = 0)
        train_y = np.concatenate((train_y, new_vals), axis = 0)
        errors = np.concatenate((errors, new_errs), axis = 0)

        surrogate.fit(train_p, train_y, errors**2)

        # monitor convergence

        # save results
        _, std = surrogate.predict(samples, return_std=True)
        new_L2 = L2_approx(std).mean()
        print()
        print(f"Iteration {i}")
        print(f"Precedent L2 approx value: {curr_L2}")
        print(f"Current L2 approx value: {new_L2}")
        print(f"Points in the training set: {len(train_p)}")
        print()

    # export final round, save
    n_burn = n_samples - 10
    n_samples += 100 
    # sample posterior
    new_samples = approx_posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

np.save(path + "outputs/samples.npy" , samples)
# run lhs