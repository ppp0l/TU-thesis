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

def d2_model(x, dom = None) :
    
    if not dom is None :
        x = (x - dom['min'] )/ (dom['max'] - dom['min'])
    
    ret = np.zeros( (len(x), 2 ))
    
    ret[:,0] = (x[:,1] - x[:,0] +1)**2/4
    
    ret[:,1]= np.sum((x+1) + np.sin(2*x)/3, axis = 1)/4
    
    return ret

def d3_laplace(x, sensors) :
    
    x2 = np.sum(x**2, axis = 1, keepdims=True)
    sens2 = np.sum(sensors**2, axis = 1, keepdims=True) 
    xsensT = x.dot(sensors.transpose())
    dist_sq = x2 + sens2.transpose() - 2 *xsensT
    dist_sq[dist_sq <0] = 0
    dist = np.sqrt( dist_sq )

    return 1/(2*dist)

def d6_diffusion(x, sensors) :
    xp = x[:,:3]
    xn = x[:,3:]

    time = sensors[:,[3]]
    sensors = sensors[:,:3]

    xp2 = np.sum(xp**2, axis = 1, keepdims=True)
    xn2 = np.sum(xn**2, axis = 1, keepdims=True)
    sens2 = np.sum(sensors**2, axis = 1, keepdims=True)
    xp_sensT = xp.dot(sensors.transpose())
    xn_sensT = xn.dot(sensors.transpose())
    distp_sq = xp2 + sens2.transpose() - 2 *xp_sensT
    distn_sq = xn2 + sens2.transpose() - 2 *xn_sensT

    return 30*(4*math.pi*time.T)**(-3/2) * ( np.exp( - distp_sq/(4*time.T)) - np.exp( - distn_sq/(4*time.T)) ) 


def time_evolving_heat_eq(x, sensors, k = 1, ts = [1,2], dom = None) :
    dim = 3
    
    if not dom is None :
        x = 2*(x - dom['min'] )/ (dom['max'] - dom['min'])-1
    
    sol = np.ones( ( len(x), len(sensors), len(ts)))
    
    x =x.reshape( (len(x),-1, dim))
    
    dist_sq = (x-sensors)**2
    dist_sq = dist_sq.sum(axis = 2)
    
    for i,t in enumerate(ts) :
        sol[:,:, i] *= (4*math.pi*k*t)**(-3/2) * np.exp( - dist_sq/(4*k*t))
        
    return 10 * sol

def stationary_heat_eq(x, sensors, k = 1, dom = None) :
    
    if not dom is None :
        x = 2*(x - dom['min'] )/ (dom['max'] - dom['min'])-1
            
    
    source1 = x[:, :2].reshape( (len(x), -1, 2) )
    source2 = x[:, 2:].reshape( (len(x), -1, 2) )
    
    dist1 = np.linalg.norm(sensors - source1, axis = 2)
    dist2 = np.linalg.norm(sensors - source2, axis = 2)
    
    sol = np.log( dist1 ) - np.log( dist2 )
    
    sol = k*sol
    
    return 2 * sol


