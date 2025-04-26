import argparse
import gc

from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.adaBeam import Adaptive_beam
from models.lipschitz import lipschitz_regressor

from IP.priors import GaussianPrior
from IP.likelihoods import lipschitz_likelihood
from IP.posteriors import Posterior 

from AL.exp_err_red import L1_err

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed

run_type = "LHSLR"
noise = "U"

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
parser.add_argument("--dim", type=int, help="Dimension of the problem")
args = parser.parse_args()
path = args.path
dim = args.dim


# manages i/o
workflow_manager = Manager(path, dim)

# loads configuration parameters
configuration = workflow_manager.read_configuration(run_type)
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
if dim == 2 :
    forward = Adaptive_beam(path + "/data/d2/kaskade", adaptive = False)
else :
    forward = fm(dim, noise, dom = param_space)

# create prior
prior_mean = IP_config["prior_mean"]
prior_std = IP_config["prior_std"]
prior = GaussianPrior(prior_mean, prior_std, param_space)

meas_std = IP_config["measurement_std"]

# active learning parameters
training_config = configuration["training_config"]
sampling_config = configuration["sampling_config"]

default_tol = training_config["default_tol"]

n_init = training_config["n_init"]


sample_every = sampling_config["sample_every"]
points_per_it = sample_every * training_config["points_per_it"]
n_it = training_config["max_iter"]//(sample_every)

threshold = training_config["threshold"]

points = lhs(param_space["min"], param_space["max"], points_per_it*n_it+n_init)

# create surrogate
surrogate = lipschitz_regressor(dim = dim, dout = forward.dout)
train_p = points[:n_init]
train_y, errors = forward.predict(train_p, tols = default_tol * np.ones(n_init))
training_set = {
    "train_p": train_p,
    "train_y": train_y,
    "errors": errors,
}
surrogate.fit(train_p, train_y, errors)

# create approximate likelihood and posterior
approx_likelihood = lipschitz_likelihood(value, meas_std, surrogate)
posterior = Posterior(approx_likelihood, prior)

n_walkers = sampling_config["n_walkers"]
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
posterior.initialize_sampler(n_walkers, initial_pos)

FE_cost = configuration["forward_model_config"]["FE_cost"]

n_sample = sampling_config["n_sample"]
n_burn = sampling_config["n_burn"]

samples = np.array([np.zeros(dim)])

points = points[n_init:]


for i in range(n_it) :

    print("Sampling posterior...") 
    print()

    # sample posterior
    new_samples = posterior.sample_points(n_sample)

    # update sample chains
    if n_burn * n_walkers > len(samples) :
        samples = np.array([np.zeros(dim)])
    else :
        samples = samples[n_burn*n_walkers :]
    samples = np.concatenate( (samples, new_samples), axis = 0)
    print("Done.")
    print()
    shortened_samples = samples[::5]

    # monitor convergence
    W = np.sum(errors**(-FE_cost))

    _, LB, UB = surrogate.predict(shortened_samples, return_bds=True)

    curr_L1 = L1_err(LB, UB).mean()

    # save results
    workflow_manager.save_results({"W": [W], "target": [curr_L1]}, run_type)
    training_set["train_p"] = train_p
    training_set["train_y"] = train_y
    training_set["errors"] = errors
    workflow_manager.state_saver(run_type, i, W, training_set, surrogate, samples)

    print("Rietriving candidates...")
    print()
    # position problem
    new_pts = points[i*points_per_it:(i+1)*points_per_it]
    print("Done.")
    print()

    new_tols = default_tol * np.ones(len(new_pts))

    print("Evaluating model...")
    print()
    # evaluate model
    new_vals, new_errs = forward.predict(new_pts, new_tols)
    
    print("Done.")
    print()

    print("Updating surrogate...")
    print()
    # update surrogate
    train_p =  np.concatenate((train_p, new_pts), axis = 0)
    train_y = np.concatenate((train_y, new_vals), axis = 0)
    errors = np.concatenate((errors, new_errs), axis = 0)

    surrogate.fit(train_p, train_y, errors)
    print("Done.")
    print()

    _, LB, UB = surrogate.predict(shortened_samples, return_bds=True)
    new_L1 = L1_err(LB, UB).mean()


    print()
    print(f"Iteration {i}")
    print(f"Precedent L2 approx value: {curr_L1}")
    print(f"Current L2 approx value: {new_L1}")
    print(f"Points in the training set: {len(train_p)}")
    print()

    if new_L1 < threshold :
        print("Convergence reached.")
        break
    gc.collect()

# export final round, save
n_burn = n_walkers * n_sample
n_sample = 2* n_sample
# sample posterior
new_samples = posterior.sample_points(n_sample)

# update sample chains
samples = samples[n_burn :]
samples = np.concatenate( (samples, new_samples), axis = 0)

shortened_samples = samples[::5]

# monitor convergence
W = np.sum(errors**(-FE_cost))

_, LB, UB = surrogate.predict(shortened_samples, return_bds=True)
curr_L1 = L1_err(LB, UB).mean()

# save results
workflow_manager.save_results({"W": [W], "target": [curr_L1]}, run_type)
workflow_manager.state_saver(run_type, n_it, W, training_set, surrogate, samples)