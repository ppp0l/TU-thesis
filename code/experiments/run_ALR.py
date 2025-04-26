import argparse

from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.lipschitz import lipschitz_regressor

from IP.priors import GaussianPrior
from IP.likelihoods import lipschitz_likelihood
from IP.posteriors import Posterior 

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from experiments.run import run

run_type = "ALR"
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
forward = fm(dim, noise, dom = param_space)

# create prior
prior_mean = IP_config["prior_mean"]
prior_std = IP_config["prior_std"]
prior = GaussianPrior(prior_mean, prior_std, param_space)

meas_std = IP_config["measurement_std"]

# active learning parameters
training_config = configuration["training_config"]

n_init = training_config["n_init"]

default_tol = training_config["default_tol"]

# create surrogate
surrogate = lipschitz_regressor(dim = dim, dout = forward.dout)
train_p = lhs(param_space["min"], param_space["max"], n_init)
train_y, errors = forward.predict(train_p, tols = default_tol * np.ones(n_init))
training_set = {
    "train_p": train_p,
    "train_y": train_y,
    "errors": errors,
}
surrogate.fit(train_p, train_y, errors)


# create approximate likelihood and posterior
approx_likelihood = lipschitz_likelihood(value, meas_std, surrogate)
approx_posterior = Posterior(approx_likelihood, prior)


sampling_config = configuration["sampling_config"]
n_walkers = sampling_config["n_walkers"]
initial_pos = lhs(param_space["min"], param_space["max"], n_walkers)
approx_posterior.initialize_sampler(n_walkers, initial_pos)

run(run_type, training_set, surrogate, forward, approx_posterior, workflow_manager)