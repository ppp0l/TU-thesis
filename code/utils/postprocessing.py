import os

import numpy as np

from models.forward import forward_model

from IP.priors import GaussianPrior
from IP.likelihoods import base_likelihood
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

