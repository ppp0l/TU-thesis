#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:48:55 2024

@author: pvillani
"""
import numpy as np
from scipy.stats import norm

class base_likelihood() :
    
    def __init__(self, y_m, std, model):
        
        self.y_m = y_m
        self.std = std
        self.var = std**2
        self.model = model

    def prob(self, p) :
        return self.plugin(p)
    
    def log_prob(self, p) :
        return self.log_plugin(p)

    def plugin(self, p):
        """
        p (np.array): must be of shape (n_pts, dim).

        """
        
        pred = self.model.predict(p)
        
        return self.plugin_pred(pred)
    
    def log_plugin(self, p) :
        
        pred = self.model.predict(p)
        
        return self.log_plugin_pred(pred)
    
    def plugin_pred(self, pred):
        """
        p (np.array): must be of shape (n_pts, dim).

        """
        
        return np.exp(- np.sum( (pred - self.y_m)**2/self.var, axis = 1) /2 )
    
    def log_plugin_pred(self, pred) :
        
        return - np.sum( (pred - self.y_m)**2/self.var, axis = 1) /2 
    

    
class lipschitz_likelihood(base_likelihood) :

    def prob(self, p) :
        return self.marginal(p)
    
    def log_prob(self, p) :
        return self.log_marginal(p)

        
    def marginal(self, p) :
        
        pred, lbd, ubd = self.model.predict(p, return_bds = True)
        
        return self.marginal_pred(pred, lbd, ubd)
    
    def log_marginal(self, p) :
        
        pred, lbd, ubd = self.model.predict(p, return_bds = True)
        
        return self.log_marginal_pred(pred, lbd, ubd)
    
    def marginal_pred(self, pred, lbd, ubd) :
        
        prob = 1/np.prod(ubd-lbd, axis = 1)
        for dim, var in enumerate(self.var) :
            prob *= norm.cdf(ubd[:,dim], loc = self.y_m [dim], scale = np.sqrt(var) ) - norm.cdf(lbd[:,dim], loc = self.y_m[dim] , scale = np.sqrt(var) )
            
        return prob
    
    def log_marginal_pred(self, pred, lbd, ubd) :
        
        return np.log(self.marginal_pred(pred, lbd, ubd))
    
    
class GP_likelihood(base_likelihood) :

    def prob(self, p) :
        return self.marginal(p)
    
    def log_prob(self, p) :
        return self.log_marginal(p)
    
    def marginal(self, p) :
        
        pred, std = self.model.predict(p, return_std = True)
        
        return self.marginal_pred(pred, std)
    
    def log_marginal(self, p) :
        
        pred, std = self.model.predict(p, return_std = True)
        
        return self.log_marginal_pred(pred, std)
    
    def marginal_pred(self, pred, std) :
        
        cov = self.var + std**2
        det = np.linalg.det(np.array( [np.diag(cov_i) for cov_i in cov] ))
        
        return np.exp(- np.sum( (pred - self.y_m)**2/cov, axis = 1) /2 ) / np.sqrt(det)
    
    def log_marginal_pred(self, pred, std) :
        
        cov = self.var + std**2
        det = np.linalg.det(np.array( [np.diag(cov_i) for cov_i in cov] ))
        
        return - np.sum( (pred - self.y_m)**2/cov, axis = 1) /2  - np.log( np.sqrt(det))

