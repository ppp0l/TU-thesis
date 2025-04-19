#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:16:38 2024

@author: pvillani
"""
import os
import random

import torch
import numpy as np
from itertools import product

from skopt.sampler import Lhs
from skopt.space import Space

from scipy.linalg import sqrtm

__all__ = ['reproducibility_seed', 
           'projection_on_simplex', 
           'latin_hypercube_sampling',
           'discretization_error_matrix'
           ]

def reproducibility_seed(seed: int) -> None:
    """To set seed for random variables."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def discretization_error_matrix(sensors, sigma=2):
    sens_sq = np.sum(sensors**2, axis = 1, keepdims=True)
    sens_sensT = sensors.dot(sensors.T)
    dist_sq = sens_sq + sens_sq.T - 2 * sens_sensT

    cov = np.exp(-sigma*dist_sq)

    std = sqrtm(cov)
    return std
    
    
def norm_sq(x, mat, pos = True):
    """
    Helper function to evaluate norm of vector w.r.t SPD matrix
    
    Args:
        x (numpy.ndarray): vector. Can be a batch.
            
        mat (numpy.ndarray): SPD matrix.
            Can be a single matrix or a vector of matrices, one for each vector.
            
        pos (bool, optional): Whether the matrix should be inverted or not. Defaults to True.
        
    Returns:
        numpy.ndarray : norm of each vector in x, w.r.t mat or mat^-1.
    """
    
    x = np.array(x)
    
    if len(x.shape) == 1 :
        
        x = x.reshape((1, len(x)))
        
    if len (mat.shape)< 3 :
        
        if pos :
            norm = np.diag(x @ mat @ x.T)
        else :
            
            solv = np.linalg.solve(mat, x.T)
            
            norm = np.diag( x @ solv)
    else:
        if pos :
            
            norm = np.array([ x[i ] @ mat[i] @ x[i ].T for i in range(len(x)) ])
            
        else :
            
            norm = np.array([ x[i ] @ np.linalg.solve( mat[i], x[i ].T ) for i in range(len(x)) ])
            
            
    if len(x) == 1 :
        return norm[0]
    
    return norm


def get_vertices( domain : dict, get_midpoints : bool = False) :
    """
    
    Auxiliary function to find vertices and eventually edges in domain.
    Not much of a smart solution, probably needs to be changed.
    
    Args:
        domain (dict): dictionary representing rectangular domain.
        
        get_edges (bool, optional): return midpoints or not. Defaults to False.
        
    Returns:
        pts (np.ndarray): list of vertices and midpoints.
    """
    
    mins = domain['min']
    maxs = domain['max']
    
    dim = len(mins)
    
    if get_midpoints :
        
        pts = np.array( [ p for p in product([0, 1/2, 1], repeat = dim)])
        
    else :
        pts = np.array( [ p for p in product([0, 1], repeat = dim)])        
    
    
    pts = pts * (maxs - mins) + mins
    return pts

def projection_on_simplex(y, k, tol=0.0001, max_iter=1000):
    func = lambda x: np.sum(np.maximum(y - x, 0)) - k
    lower = np.min(y) - k / len(y)
    upper = np.max(y)
    
    for _ in range(max_iter):
        midpoint = (upper + lower) / 2.0
        value = func(midpoint)
    
        if (len(y) *abs(value)/k) <= tol:
            break
    
        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint
    
    return np.maximum(y - midpoint, 0)

def latin_hypercube_sampling(
    domain_lower_bound: np.ndarray,
    domain_upper_bound: np.ndarray,
    n_sample: int,
    method="maximin",
):
    domain = np.vstack((domain_lower_bound, domain_upper_bound)).astype(float).T
    space = Space(list(map(tuple, domain)))
    lhs = Lhs(criterion=method, iterations=5000)
    samples = lhs.generate(space.dimensions, n_sample)

    return np.array(samples, dtype=float)

class Scaler :
    # Class used to scale parameters and function values to make them more digestable for the GPR
    #
    # Everything is done by calling Scaler(input)
    # Behaviour defined by the inputs given and the Scaler.output switch
    
    
    def __init__(self, par_dims=2, fun_dims = 3, isTorch=True):
        
        # This is a switch for the mode:
        # if False, the Scaler trains itself and builds the transformations
        # if True, the Scaler scales the given vectors (params or function values)
        self.output= False 
        self.isTorch = isTorch
        
        linear_par = torch.ones(par_dims) #Linear part of the params transf.
        affine_par = torch.zeros(par_dims) # Affine part of the params transf.
        
        linear_fun = torch.eye(fun_dims) #Linear part of the values transf.
        # We have no affine part for function values since we are interested in scaling only
        
        # Transformation on the param. space
        self.par_transf = { "lin" : linear_par,
                        "aff" : affine_par,
                       }
        # Transformation of the function values
        self.fun_transf = { "lin" : linear_fun,
                         }
        
    
    # __call__ can:
    # 1) initialise the transformations (self.output=False)
    # 2) scale any vector once turned in output mode
    #
    # In output mode, the differentiation between params and function values is done by input:
    # 1) To scale params you have to pass par
    # 2) To scale function values you have to pass fun
    # 
    # In input mode:
    # 1) to set the params transformation you pass a space
    # 2) to set the function values transf you pass a fun containing function values
    #
    # types: 
    # - space : dictionary with keys "min" and "max" (minima and maxima of the param space)
    # - fun, par : array or tensors of size Neval * dim of codomain/space
    
    def __call__(self, space = None, fun = None, par = None):
        
        
        
        if not self.output :
            if (space is None)==False :
                
                return self.set_par_transf(space)
            
            if (fun is None)==False :
                
                if not (torch.is_tensor(fun)):
                    fun = torch.tensor(fun)
                    
                return self.set_fun_transf(fun)
            
        else :
            if (par is None)==False :
                
                if not (torch.is_tensor(par)):
                    par = torch.tensor(par)
                    
                return self.par_scale( par )
            
            if (fun is None)==False  :
                
                if not (torch.is_tensor(fun)):
                    fun = torch.tensor(fun)
                    
                
                return self.fun_scale(fun)
            
        return 0
                    
    def set_par_transf(self, space) :
        parMax = torch.tensor(space["max"])
        parMin = torch.tensor(space["min"])
        
        lin = 1 / ( parMax - parMin )
        aff = - parMin * lin 
        
        self.par_transf["lin"] = lin
        self.par_transf["aff"] = aff
        
        scaledMax = self.par_scale(parMax)
        scaledMin = self.par_scale(parMin)
        
        return { "max" : scaledMax,
                 "min" : scaledMin
               }
    
    def set_fun_transf(self, vec) :
        mean= torch.norm(vec, dim=0, p= 1) / len(vec)
        
        mean[mean==0]=1
        
        self.fun_transf["lin"] = 1/mean
        return self.fun_scale(vec)
        
        
    def par_scale(self, vec) :
        vec = vec * self.par_transf["lin"] + self.par_transf["aff"]
        
        if not self.isTorch :
            vec = vec.numpy()
        
        return vec
        
        
    def fun_scale(self, vec) :
        vec = vec * self.fun_transf["lin"]
        
        if not self.isTorch :
            vec = vec.numpy()
        
        return vec
        
    def rescale(self, fun = None, par = None) :
        
        if (fun is None) == False :
            
            if not (torch.is_tensor(fun)):
                fun = torch.tensor(fun)
                    
            fun = fun / self.fun_transf["lin"]

            if not self.isTorch :
                fun = fun.numpy()

            return fun
        
        if (par is None) == False :
            
            if not (torch.is_tensor(par)):
                par = torch.tensor(par)
                
            par = (par - self.par_transf["aff"] ) / self.par_transf["lin"]
            
            if not self.isTorch :
                par = par.numpy()
                
            return par
            
            
    
    def input_(self) :
        self.output= False
        
    def output_(self) :
        self.output= True