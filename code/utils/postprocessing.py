import os

import numpy as np

from models.forward import forward_model
from models.GP_models.MTSurrogate import MTModel
from models.lipschitz import lipschitz_regressor

from IP.priors import GaussianPrior
from IP.likelihoods import base_likelihood, GP_likelihood, lipschitz_likelihood
from IP.posteriors import Posterior

from utils.utils import latin_hypercube_sampling as lhs

import json
import torch

def gt_samples(dim, config) :
    IP_config = config["IP_config"]

    domain = {
        "min": IP_config["domain_lower_bound"],
        "max": IP_config["domain_upper_bound"]
    }

    prior = GaussianPrior(
        mean=IP_config["prior_mean"],
        std=IP_config["prior_std"],
        dom= domain,
    )
    
    forward = forward_model(dim=dim, dom=domain)

    likelihood = base_likelihood(
        y_m = IP_config["measurement"],
        std=IP_config["measurement_std"],
        model=forward,
    )

    true_posterior = Posterior(
        prior=prior,
        likelihood=likelihood,
    )
    n_walkers = config["sampling_config"]["n_walkers"]
    starting_pos = lhs( domain["min"], domain["max"], n_walkers)
    true_posterior.initialize_sampler(n_walkers, starting_pos)

    return true_posterior.sample_points(1000)

def gt_MAP(dim, config, starts = None) :
    IP_config = config["IP_config"]

    domain = {
        "min": IP_config["domain_lower_bound"],
        "max": IP_config["domain_upper_bound"]
    }

    prior = GaussianPrior(
        mean=IP_config["prior_mean"],
        std=IP_config["prior_std"],
        dom= domain,
    )
    
    forward = forward_model(dim=dim, dom=domain)

    likelihood = base_likelihood(
        y_m = IP_config["measurement"],
        std=IP_config["measurement_std"],
        model=forward,
    )

    true_posterior = Posterior(
        prior=prior,
        likelihood=likelihood,
    )
    
    return true_posterior.MAP_estimate(starts) 

def compute_surrogate_MAP(config, starts, type_surr, training_set, state_dict, cov_est = None) :

    IP_config = config["IP_config"]

    domain = {
        "min": IP_config["domain_lower_bound"],
        "max": IP_config["domain_upper_bound"]
    }

    prior = GaussianPrior(
        mean=IP_config["prior_mean"],
        std=IP_config["prior_std"],
        dom= domain,
    )

    training_p = training_set['train_p']
    training_y = training_set['train_y']
    errors = training_set['errors']

    if type_surr == "GP" :
        surrogate = MTModel(num_tasks = len(training_y[0]))
        likelihood_class = GP_likelihood

    elif type_surr == "LR" :
        surrogate = lipschitz_regressor( dim = len(training_p[0]), dout = len(training_y[0]))
        likelihood_class = lipschitz_likelihood
                   
    if cov_est is None :
        surrogate.fit(training_p, training_y, errors)
    else :
        surrogate.fit(training_p, training_y, errors, likelihood_has_task_noise=True, likelihood_task_noise=torch.tensor(cov_est))
    
    surrogate.load_state_dict(state_dict)

    likelihood = likelihood_class(
            y_m = IP_config["measurement"],
            std=IP_config["measurement_std"],
            model=surrogate,
        )
    
    posterior = Posterior(
        prior=prior,
        likelihood=likelihood,
    )

    return posterior.MAP_estimate(starts)


