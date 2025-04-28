"""
created on: 2025/01/19

@author: pvillani
"""
import argparse

from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.adaBeam import Adaptive_beam
from models.GP_models.MTSurrogate import MTModel

from IP.priors import GaussianPrior
from IP.likelihoods import GP_likelihood
from IP.posteriors import Posterior 

from AL.cov_est import estimate_covariance

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from experiments.run import run

run_type = "AGP"
noise = "N"

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

# create prior
prior_mean = IP_config["prior_mean"]
prior_std = IP_config["prior_std"]
prior = GaussianPrior(prior_mean, prior_std, param_space)

meas_std = IP_config["measurement_std"]

# active learning parameters
training_config = configuration["training_config"]

n_init = training_config["n_init"]

default_tol = training_config["default_tol"]

# create forward model,  sets noise type
if dim == 2 :
    forward = Adaptive_beam(path + "/data/d2/kaskade", adaptive = True, default_tol= default_tol)
    forward.dom = param_space
else :
    forward = fm(dim, noise, dom = param_space)

# create surrogate
surrogate = MTModel(num_tasks = forward.dout)
train_p = lhs(param_space["min"], param_space["max"], n_init)
train_y, errors = forward.predict(train_p, tols = default_tol * np.ones(n_init))

residuals, tolerances = forward.get_residuals()
eval_cov = estimate_covariance(residuals, tolerances)

training_set = {
    "train_p": train_p,
    "train_y": train_y,
    "errors": errors,
}
surrogate.fit(train_p, train_y, errors, likelihood_has_task_noise=True, likelihood_task_noise=eval_cov)

# create approximate likelihood and posterior
approx_likelihood = GP_likelihood(value, meas_std, surrogate)
approx_posterior = Posterior(approx_likelihood, prior)


sampling_config = configuration["sampling_config"]
n_walkers = sampling_config["n_walkers"]
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
approx_posterior.initialize_sampler(n_walkers, initial_pos)

run(run_type, training_set, surrogate, forward, approx_posterior, workflow_manager)