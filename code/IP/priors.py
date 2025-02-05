import numpy as np
from scipy.stats import multivariate_normal

class Prior:
    def prob(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    def log_prob(self, x):
        raise NotImplementedError("Subclasses should implement this!")

class FlatPrior(Prior):
    def __init__(self, dom):
        self.dom = dom
        self.dim = len(dom['min'] )
        self.area = np.prod(dom["max"] - dom["min"] )

    def prob(self, p):
        p = np.reshape(p, (-1, self.dim))
        in_dom = np.array( (p > self.dom['min']) * (p < self.dom['max']), dtype = bool )
        if self.dim > 1:
            in_dom = in_dom.all(axis = 1)

        in_dom = in_dom.reshape( (-1,))
        return in_dom / self.area

    def log_prob(self, p):
        return np.log(self.prob(p))

class GaussianPrior(Prior):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.rv = multivariate_normal(mean=mean, cov=cov)

    def prob(self, x):
        return self.rv.pdf(x)

    def log_prob(self, x):
        return self.rv.logpdf(x)