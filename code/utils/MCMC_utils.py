# -*- coding: utf-8 -*-

import numpy as np
import math
import emcee

try:
    from scipy.special import logsumexp
    from scipy.stats import multivariate_t
except ImportError:
    multivariate_t = None

from emcee.moves import RedBlueMove

__all__ = ["DIMEMove", 
           'burn_in',
           'MCMCError', 
           ]
    
class MCMCError(Exception) :
    pass


def burn_in(MCMC_sampler, start, n_burn = None, **kwargs) :
    dim = len(start[0])
    if isinstance(start, emcee.State) :
        n_walkers = len(start.coords)
    else :
        n_walkers = len(start)
    if n_burn is None:
        n_burn = 100
    
    # run some hundreds of samples
    state = MCMC_sampler.run_mcmc(start, n_burn, tune= True, **kwargs)
    chains = MCMC_sampler.get_chain()
    
    # check convergence of chains with Gelman Rubin R hat
    N = n_burn // 2
    M = n_walkers * 2
    # split chains in half to improve convergence check, as in mcstan docs
    resh = np.zeros( (n_burn//2, M, dim) )
    for  i in range(M) :
        
        resh[:,i] = chains[n_burn//2 * (i%2) : n_burn //2* (i%2 +1)  , i//2]
    chains = resh    
    
    mean = chains.mean(axis = (0,1) )
    chain_mean = chains.mean(axis = 0)
    
    between_chain_var = 1/(M  -1) * np.sum( (chain_mean - mean)**2, axis = (0,1) )
    within_chain_var = 1/(N - 1) *np.sum( (chains - chain_mean)**2, axis = 0  )
    average_within_var = np.mean( within_chain_var, axis = 0 )
    
    var_est = between_chain_var + (N -1)/ N * average_within_var
    
    R = np.sqrt( var_est/average_within_var)
    
    # print(f'R: {R}')
    # print(f'chain means : {chain_mean}')
    # print(f'within chain : {average_within_var}')
    # print(f'between chain ; {between_chain_var}')
    # print()
    
    # if chains converged return
    if np.all(R<1.1):
        MCMC_sampler.reset()
        return state, True
    
    # if not converged, reinitialise chains around maxima
    samples =  MCMC_sampler.get_chain( flat = True)
    log_prob =  MCMC_sampler.get_log_prob( flat = True)
    
    n_discard = math.ceil( n_burn* (1 - 2 / n_walkers)) 
    #n_keep = n_burn * n_walkers
    if isinstance(start, emcee.State) :
        state = np.zeros_like(start.coords)
    else :
        state = np.zeros_like(start)
        
    var = np.ones_like(state)
    # find maxima
    for i in range( n_walkers) :
        
        #n_keep = n_keep - n_discard
        
        indmax = np.argmax(log_prob)
        
        state[i] = samples[indmax]
        
        dist = np.linalg.norm( (samples - samples[indmax]), ord = 2, axis = 1)
        
        ordered_args = np.argpartition( dist, n_discard)
        
        close = ordered_args [:n_discard]
        
        var[i] = np.mean( ( state[i] - samples[close] )**2 , axis = 0 ) 
            
        not_so_close = ordered_args[n_discard:]
        
        log_prob = log_prob[not_so_close]
        samples = samples[not_so_close]
        
        
    MCMC_sampler.reset()
    
    # perturb maxima a bit
    state = state + np.random.normal(loc =0.0, scale = 2*np.sqrt(var), size = state.shape )
    
    # print(state)
    # print()
    return state, False


def mvt_sample(df, mean, cov, size, random):
    """Sample from multivariate t distribution

    The results from random.multivariate_normal with non-identity covariance matrix are not reproducibel across OS architecture. Since scipy.stats.multivariate_t is based on numpy's multivariate_normal, the workaround is to provide this manually.
    """

    dim = len(mean)

    # draw samples
    snorm = random.randn(size, dim)
    chi2 = random.chisquare(df, size) / df

    # calculate sqrt of covariance
    svd_cov = np.linalg.svd(cov * (df - 2) / df)
    sqrt_cov = svd_cov[0] * np.sqrt(svd_cov[1]) @ svd_cov[2]

    return mean + snorm @ sqrt_cov / np.sqrt(chi2)[:, None]


class DIMEMove(RedBlueMove):
    r"""A proposal using adaptive differential-independence mixture ensemble MCMC.

    This is the `Differential-Independence Mixture Ensemble proposal` as developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/dime_mcmc_boehl.pdf>`_.

    Parameters
    ----------
    sigma : float, optional
        standard deviation of the Gaussian used to stretch the proposal vector.
    gamma : float, optional
        mean stretch factor for the proposal vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
    aimh_prob : float, optional
        probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to :math:`0.1`.
    df_proposal_dist : float, optional
        degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to :math:`10`.
    rho : float, optional
        decay parameter for the aimh proposal mean and covariances. Defaults to :math:`0.999`.
    """

    def __init__(
        self,
        sigma=1.0e-5,
        gamma=None,
        aimh_prob=0.1,
        df_proposal_dist=10,
        rho=0.999,
        **kwargs
    ):
        if multivariate_t is None:
            raise ImportError(
                "you need scipy.stats.multivariate_t and scipy.special.logsumexp to use the DIMEMove"
            )

        self.sigma = sigma
        self.g0 = gamma
        self.decay = rho
        self.aimh_prob = aimh_prob
        self.dft = df_proposal_dist

        super(DIMEMove, self).__init__(**kwargs)

    def setup(self, coords):
        # set some sane defaults

        nchain, npar = coords.shape

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if not hasattr(self, "prop_cov"):
            # even more MAGIC
            self.prop_cov = np.eye(npar)
            self.prop_mean = np.zeros(npar)
            self.accepted = np.ones(nchain, dtype=bool)
            self.cumlweight = -np.inf
        else:
            # update AIMH proposal distribution
            self.update_proposal_dist(coords)

    def propose(self, model, state):
        """Wrap original propose to get the some info on the current state"""

        self.lprobs = state.log_prob
        state, accepted = super(DIMEMove, self).propose(model, state)
        self.accepted = accepted
        return state, accepted

    def update_proposal_dist(self, x):
        """Update proposal distribution with ensemble `x`"""

        nchain, npar = x.shape

        # log weight of current ensemble
        if self.accepted.any():
            lweight = (
                logsumexp(self.lprobs)
                + np.log(sum(self.accepted))
                - np.log(nchain)
            )
        else:
            lweight = -np.inf

        # calculate stats for current ensemble
        ncov = np.cov(x.T, ddof=1)
        nmean = np.mean(x, axis=0)

        # update AIMH proposal distribution
        newcumlweight = np.logaddexp(self.cumlweight, lweight)
        self.prop_cov = (
            np.exp(self.cumlweight - newcumlweight) * self.prop_cov
            + np.exp(lweight - newcumlweight) * ncov
        )
        self.prop_mean = (
            np.exp(self.cumlweight - newcumlweight) * self.prop_mean
            + np.exp(lweight - newcumlweight) * nmean
        )
        self.cumlweight = newcumlweight + np.log(self.decay)

    def get_proposal(self, x, xref, random):
        """Actual proposal function"""

        xref = xref[0]
        nchain, npar = x.shape
        nref, _ = xref.shape

        # differential evolution: draw the indices of the complementary chains
        i0 = np.arange(nchain) + random.randint(1, nchain, size=nchain)
        i1 = np.arange(nchain) + random.randint(1, nchain - 1, size=nchain)
        i1 += i1 >= i0
        # add small noise and calculate proposal
        f = self.sigma * random.randn(nchain)
        q = x + self.g0 * (xref[i0 % nref] - xref[i1 % nref]) + f[:, None]
        factors = np.zeros(nchain, dtype=np.float64)

        # draw chains for AIMH sampling
        xchnge = random.rand(nchain) <= self.aimh_prob

        # draw alternative candidates and calculate their proposal density
        xcand = mvt_sample(
            df=self.dft,
            mean=self.prop_mean,
            cov=self.prop_cov,
            size=sum(xchnge),
            random=random,
        )
        lprop_old, lprop_new = multivariate_t.logpdf(
            np.vstack((x[None, xchnge], xcand[None])),
            self.prop_mean,
            self.prop_cov * (self.dft - 2) / self.dft,
            df=self.dft,
        )

        # update proposals and factors
        q[xchnge, :] = np.reshape(xcand, (-1, npar))
        factors[xchnge] = lprop_old - lprop_new

        return q, factors
