"""
created on: 2025/01/19

@author: pvillani
"""
import argparse
import gc

from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.GP_models.MTSurrogate import MTModel

from IP.priors import GaussianPrior
from IP.likelihoods import base_likelihood, GP_likelihood
from IP.posteriors import Posterior 

from AL.L2_GP import L2_approx
from AL.position.GP import solve_pos_prob
from AL.tolerance.GP import solve_acc_prob

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from utils.plots import corner_plot

dim = 3

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
path = args.path

# manages i/o
workflow_manager = Manager(path, dim)

# loads configuration parameters
configuration = workflow_manager.read_configuration()
IP_config = configuration["IP_config"]

value = IP_config["measurement"]
ground_truth = IP_config["ground_truth"]

# sets seed
reproducibility_seed(seed = configuration["seed"])

### actual task

# parameter space
domain_UB = IP_config["domain_upper_bound"]
domain_LB = IP_config["domain_lower_bound"]
param_space = { 
    'min' : domain_LB, 
    'max' : domain_UB,
}

# create forward model,  sets noise type
forward = fm(dim, "N", dom = param_space)

# create prior
mean = IP_config["prior_mean"]
std = IP_config["prior_std"]
prior = GaussianPrior(mean, std, param_space)

# create true likelihood
meas_std = IP_config["measurement_std"]
true_likelihood = base_likelihood(value, meas_std, forward)

# create true posterior
true_posterior = Posterior(true_likelihood, prior)

# active learning parameters
training_config = configuration["training_config"]
sampling_config = configuration["sampling_config"]

n_init = training_config["n_init"]
points_per_it = training_config["points_per_it"]
n_it = training_config["n_it"]
sample_every = sampling_config["sample_every"]

default_tol_fixed = training_config["default_tol_fixed"]
default_tol_ada = training_config["default_tol_ada"]
budget_ada = training_config["budget"]
budget_per_it = budget_ada / n_it

FE_cost = configuration["forward_model_config"]["FE_cost"]

# create surrogate
surrogate = MTModel(num_tasks = forward.dout)
train_p = lhs(param_space["min"], param_space["max"], n_init)
train_y, errors = forward.predict(train_p, tols = default_tol_ada * np.ones(n_init))
surrogate.fit(train_p, train_y, errors**2)
# export initial surrogate

# create approximate likelihood and posterior
approx_likelihood = GP_likelihood(value, meas_std, surrogate)
approx_posterior = Posterior(approx_likelihood, prior)

n_walkers = sampling_config["n_walkers"]
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
approx_posterior.initialize_sampler(n_walkers, initial_pos)

n_samples = sampling_config["init_samples"]
samples = np.array([np.zeros(dim)])

true_posterior.initialize_sampler(n_walkers, initial_pos)
true_samples = true_posterior.sample_points(4000)

tr_mean = np.mean(true_samples, axis = 0)
tr_std  = np.sqrt( np.mean(true_samples**2, axis = 0 )- tr_mean**2)
cleaned_true = true_samples[np.all( true_samples - tr_mean < 4*tr_std, axis = 1)]

for i in range(n_it) :
    if i%sample_every == 0 :
        print("Sampling posterior...") 
        print()
        n_burn = n_samples - 30
        n_samples += 80
        # sample posterior
        new_samples = approx_posterior.sample_points(n_samples)

        # update sample chains
        samples = samples[n_burn :]
        samples = np.concatenate( (samples, new_samples), axis = 0)
        print("Done.")
        print()

    samp_mean = np.mean(samples, axis = 0)
    samp_std  = np.sqrt( np.mean(samples**2, axis = 0 )- samp_mean**2)
    cleaned_samples = samples[np.all( samples - samp_mean < 4*samp_std, axis = 1)]

    corner_plot(
        [cleaned_samples, cleaned_true[:len(cleaned_samples)]], 
        colors = ["teal", "crimson"],
        labels = ["AGP approximation", "Ground truth"],
        points = [train_p],
        points_colors = ["black"],
        title = f"Samples at iteration {i}",
        domain = param_space,
        savepath = configuration["res_path"] + f"/samples_{i}.png",
    )

    _, std = surrogate.predict(samples, return_std=True)

    curr_L2 = L2_approx(std).mean()

    print("Rietriving candidates...")
    print()
    # position problem
    candidates = solve_pos_prob(points_per_it, param_space, default_tol_ada, surrogate, samples, std, FE_cost )
    print("Done.")
    print()

    print("Optimizing tolerances...")
    print()
    # accuracy problem
    tolerances, new_pts, updated = solve_acc_prob(candidates, budget_per_it, surrogate, samples, std, FE_cost)
    print("Done.")
    print()

    new_tols = tolerances[-len(new_pts):]
    update_tols = tolerances[:-len(new_pts)]

    print("Evaluating model...")
    print()
    # evaluate model
    if len(new_pts) > 0:
        new_vals, new_errs = forward.predict(new_pts, new_tols)
    if np.any(updated) :
        train_y[updated], errors[updated] = forward.predict(train_p[updated], update_tols[updated])

    print("Done.")
    print()

    print("Updating surrogate...")
    print()
    # update surrogate
    train_p =  np.concatenate((train_p, new_pts), axis = 0)
    train_y = np.concatenate((train_y, new_vals), axis = 0)
    errors = np.concatenate((errors, new_errs), axis = 0)

    surrogate.fit(train_p, train_y, errors**2)
    print("Done.")
    print()


    # monitor convergence
    # save results
    W = np.sum(errors**(-FE_cost))

    _, std = surrogate.predict(samples, return_std=True)
    new_L2 = L2_approx(std**2).mean()
    workflow_manager.save_results({"W": [W], "target": [new_L2]}, "AGP")


    print()
    print(f"Iteration {i}")
    print(f"Precedent L2 approx value: {curr_L2}")
    print(f"Current L2 approx value: {new_L2}")
    print(f"Points in the training set: {len(train_p)}")
    print()

    gc.collect()



# export final round, save
n_burn = n_samples - 30
n_samples += 80
# sample posterior
new_samples = approx_posterior.sample_points(n_samples)

# update sample chains
samples = samples[n_burn :]
samples = np.concatenate( (samples, new_samples), axis = 0)

samp_mean = np.mean(samples, axis = 0)
samp_std  = np.sqrt( np.mean(samples**2, axis = 0 )- samp_mean**2)
cleaned_samples = samples[np.all( samples - samp_mean < 4*samp_std, axis = 1)]

corner_plot(
    [cleaned_samples, cleaned_true[:len(cleaned_samples)]], 
    colors = ["teal"],
    labels = ["Ground truth", "AGP approximation"],
    points = [train_p],
    points_colors = ["black"],
    title = f"Samples at iteration {i+1}",
    domain = param_space,
    savepath = configuration["res_path"] + f"/samples_{i+1}.png",
)
