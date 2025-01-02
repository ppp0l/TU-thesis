#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:31:56 2024

@author: pvillani
"""
import numpy as np
import math
            
def wiggly(x, n, dom) :
    if len(x) == 0 :
        return 0 
    x = np.reshape(x, (len(x), -1) )
    if dom is not None :
        x = (x - dom['min'] )/ (dom['max'] - dom['min']) 
    
    sin = np.sin(10*x-1)
    cos = np.cos(10*x-1)

    wobble = np.zeros( (len(x), n) )

    for i in range(n) :

        wobble[:,i] = 1/10 * (np.sin(20*x-2).sum(axis=1))# np.mean( sin * cos, axis = 1 ) /5

    return wobble

def d1_model(x) :
    ret = (1 - x + x**2 )* np.sin(4 * x + 1)
    ret =  (1+x) * np.sin(6 * x + 1)
    return ret

def d2_model(x, n = 2, dom = None) :
    
    if not dom is None :
        x = (x - dom['min'] )/ (dom['max'] - dom['min'])-1/2
    
    ret = np.zeros( (len(x), n ))
    
    diffs = x[:,0]-x[:,1]
    sums = x[:,0]+x[:,1]
    for i in range(n) :
        
        ret[:,i] = np.sin(1*i) * diffs   + np.cos(1*i) * sums 
    
    return ret

def time_evolving_heat_eq(x, sensors, k = 1, ts = [1,2], dim = 3, dom = None) :
    
    if not dom is None :
        x = 2*(x - dom['min'] )/ (dom['max'] - dom['min'])-1
    
    sol = np.ones( ( len(x), len(sensors), len(ts)))
    
    x =x.reshape( (len(x),-1, dim))
    
    dist_sq = (x-sensors)**2
    dist_sq = dist_sq.sum(axis = 2)
    
    for i,t in enumerate(ts) :
        sol[:,:, i] *= (4*math.pi*k*t)**(-3/2) * np.exp( - dist_sq/(4*k*t))
        
    return sol

def stationary_heat_eq(x, sensors, k = 1, dom = None) :
    
    if not dom is None :
        x = 2*(x - dom['min'] )/ (dom['max'] - dom['min'])-1
            
    
    source1 = x[:, :2].reshape( (len(x), -1, 2) )
    source2 = x[:, 2:].reshape( (len(x), -1, 2) )
    
    dist1 = np.linalg.norm(sensors - source1, axis = 2)
    dist2 = np.linalg.norm(sensors - source2, axis = 2)
    
    sol = np.log( dist1 ) - np.log( dist2 )
    
    sol = k*sol
    
    return sol


def forward_model(x, inp, out=None, dom=None, wiggle = True, 
                  sensors : np.ndarray = None,
                  meas_time : np.ndarray = None,
                  k = 0.5,
                  ) :
    if out is None:
        out = inp+1
        
    match inp :
        case 1 :
            val = d1_model(x, n = out, dom = dom) 
        
        case 2 : 
            val = d2_model(x, n = out, dom = dom) 
        
        case 3 :
            if x.size == 0 :
                return np.zeros((len(x), len(sensors) * len(meas_time)))
            val = time_evolving_heat_eq(x, sensors, dom = dom, ts = meas_time, k = k)
            val = val.reshape((len(x), -1) )
            wiggle = False
            
        case 4 :
            if x.size == 0 :
                return np.zeros((len(x), len(sensors)))
            
            val = stationary_heat_eq(x, sensors, dom = dom, k = k)
            val = val.reshape((len(x), -1) )
            wiggle = False
            
        case _:
            raise NotImplementedError()
            
    if wiggle : 
        val += wiggly(x, out, dom)
        
    return val
