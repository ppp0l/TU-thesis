"""
created on: 2025/01/19

@author: pvillani
"""
import argparse
from utils.workflow import Manager

import numpy as np

from models.forward import forward_model as fm
from models.surrogate import Surrogate

from AL.acquisition import pos_EER

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from utils.metrics import experror, TVD

parser = argparse.ArgumentParser()
parser.add_argument("--dim", default=2, type=int, help="Dimension of parameter space")
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
dim = args.dim
path = args.path

# manages i/o
workflow_manager = Manager(path, dim)

# loads configuration parameters
configuration = workflow_manager.read_parameters()

# sets seed
reproducibility_seed(seed = configuration['seed']+1)

### actual task

# parameter space
param_space = { 
    'min' : np.zeros(dim), 
    'max' : np.ones(dim)
}

# create forward model,  sets noise type
forward = fm(dim, configuration["noise_type"], dom = param_space)

# create true posterior
####

# active learning parameters
n_init = configuration["n_init"]
points_per_it = dim
n_it = 10


default_tol = configuration["default_tol"]
FE_cost = configuration["FE cost"]
budget = n_it * points_per_it * (default_tol)**(-FE_cost)

# create surrogate
surrogate = Surrogate(dim, dout = forward.dout)
# export initial surrogate

# create approximate posterior
####

for type_run in ["posAd","fullyAd"] :
    # load initial
    for i in range(n_it) :
        # sample posterior

        # position problem

        # accuracy problem

        # update surrogate

        # monitor convergence

        # save results
        print()
        print(f"Iteration {i}")

        print()

    # export final round, save

# run lhs