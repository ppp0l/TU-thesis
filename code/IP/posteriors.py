import numpy as np
import emcee
from IP.likelihoods import base_likelihood
from IP.priors import Prior
from utils.MCMC_utils import DIMEMove, burn_in

class Posterior:
    def __init__(self, likelihood: base_likelihood, prior : Prior):
        self.likelihood = likelihood
        self.prior = prior
        self.sampler = None
        self.sampler_start = None

    def prob(self, p) :
        if hasattr(self, "ndim") :
            dim = self.ndim
        else :
            dim = self.likelihood.model.dim
        p = p.reshape((-1, dim))
            
        return self.prior.prob(p) * self.likelihood.prob(p)

    def log_prob(self, p):
        if hasattr(self, "ndim") :
            dim = self.ndim
        else :
            dim = self.likelihood.model.dim
        p = p.reshape((-1, dim))
        
        log_prior = self.prior.log_prob(p)
        log_likelihood = self.likelihood.log_prob(p)
        return log_prior + log_likelihood

    def initialize_sampler(self, n_walkers, initial_pos):
        ndim = len(initial_pos[0])
        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_prob, moves=DIMEMove())
        self.burn_in(initial_pos)

    def burn_in(self, initial_pos):
        if self.sampler is None:
            raise ValueError("Sampler not initialized. Call initialize_sampler first.")
        burned_in = False
        nit = 0
        while not burned_in :
            if nit > 4 :
                break
            self.sampler_start, burned_in = burn_in(self.sampler, initial_pos)
            nit+=1

    def sample_points(self, n_steps):
        if self.sampler is None:
            raise ValueError("Sampler not initialized. Call initialize_sampler first.")
        if self.sampler_start is None :
            raise ValueError("Initial position of sampler not defined, burn in or provide one.")
        
        self.sampler.reset()
        
        self.sampler.run_mcmc(self.sampler_start, n_steps)
        self.sampler_start = self.sampler.get_last_sample()
        chain = self.sampler.get_chain(flat= True)

        return chain

