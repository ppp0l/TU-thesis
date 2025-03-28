"""
created on: 2025/01/19

@author: pvillani
"""
import argparse
from utils.workflow import Manager

import numpy as np

import corner

from models.forward import forward_model as fm
from models.lipschitz import lipschitz_regressor

from IP.priors import FlatPrior
from IP.likelihoods import base_likelihood, lipschitz_likelihood
from IP.posteriors import Posterior 

from AL.exp_err_red import L1_err
from AL.position.lips import solve_pos_prob
from AL.tolerance.lips import solve_acc_prob

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
value = np.array([0.33069597382962845, 0.23476875535969113, 0.13201256840100192, -0.04325756645644445, -0.18841948370375194, 
                  -0.37610577847838084, -0.1493904551279874, -0.4478949378911753, -0.1410582844824224, -0.15266744400879384, 
                  0.01789137267732835, 0.1927816879708893])# np.array([0.36768606, 0.83379858])

# sets seed
reproducibility_seed(seed = 17)

### actual task

# parameter space
param_space = { 
    'min' : np.zeros(dim), 
    'max' : np.ones(dim)
}

# create forward model,  sets noise type
forward = fm(dim, "U", dom = param_space)

# create prior
prior = FlatPrior(param_space)

# create true likelihood
meas_cov = 0.1 * np.ones(forward.dout)
true_likelihood = base_likelihood(value, meas_cov, forward)

# create true posterior
true_posterior = Posterior(true_likelihood, prior)


# active learning parameters
n_init = 20
points_per_it = 1
n_it = 40


default_tol = 0.02
FE_cost = 1
budget = points_per_it * (default_tol)**(-FE_cost)

# create surrogate
surrogate = lipschitz_regressor(dim, forward.dout)
train_p = lhs(param_space["min"], param_space["max"], n_init)
train_y, errors = forward.predict(train_p, tols = default_tol * np.ones( (n_init,forward.dout)))
surrogate.fit(train_p, train_y, errors)
# export initial surrogate

# create approximate likelihood and posterior
approx_likelihood = lipschitz_likelihood(value, meas_cov, surrogate)
approx_posterior = Posterior(approx_likelihood, prior)

n_walkers = 16
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
approx_posterior.initialize_sampler(n_walkers, initial_pos)

true_posterior.initialize_sampler(n_walkers, initial_pos)
true_samples = true_posterior.sample_points(4000)

n_samples = 0
samples = np.array([np.zeros(dim)])
for type_run in ["fullyAd"] :
    # load initial
    for i in range(n_it) :
        if i%dim == 0 :
            n_burn = n_samples - 10
            n_samples += 50 
            # sample posterior
            new_samples = approx_posterior.sample_points(n_samples)

            # update sample chains
            samples = samples[n_burn*n_walkers:]
            samples = np.concatenate( (samples, new_samples), axis = 0)

        _, LB, UB = surrogate.predict(samples, return_bds=True)

        curr_L1 = L1_err(LB, UB).mean()

        # position problem
        candidates = solve_pos_prob(points_per_it, param_space, default_tol, surrogate, samples)

        # accuracy problem
        tolerances, new_pts, updated = solve_acc_prob(candidates, budget, surrogate, samples, FE_cost)

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
        _, LB, UB = surrogate.predict(samples, return_bds=True)
        new_L1 = L1_err(LB, UB).mean()
        print()
        print(f"Iteration {i}")
        print(f"Precedent L1 approx value: {curr_L1}")
        print(f"Current L1 approx value: {new_L1}")
        print(f"Points in the training set: {len(train_p)}")
        print()

        if i%5== 0 :
            fig = corner.corner(samples, color="crimson", plot_datapoints = False)
            corner.corner(true_samples[:len(samples)], color="teal", plot_datapoints = False, fig = fig)
            corner.overplot_points(fig, train_p, color="black", markersize = 5)
            fig.savefig(path + f"/outputs/samples_{i}.png")
            #fig.close()

    # export final round, save
    n_burn = n_samples - 10
    n_samples += 100 
    # sample posterior
    new_samples = approx_posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

    fig = corner.corner(samples, color="crimson", plot_datapoints = False)
    corner.corner(true_samples[:len(samples)], color="teal", plot_datapoints = False, fig = fig)
    corner.overplot_points(fig, train_p, color="black", markersize = 5 )
    fig.savefig(path + f"/outputs/samples_{i}.png")
    #fig.close()
    


np.save(path + "/outputs/samples.npy" , samples)
# run lhs