#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:06:30 2024

@author: pvillani
"""
import numpy as np

def experror(mod1, mod2, dens) :
    
    grid = np.linspace(0,1,1000).reshape((-1,1))
    
    distr = dens(grid)
    distr = distr/distr.mean()
    
    res1 = mod1.predict(grid)
    res2 = mod2.predict(grid)
    
    return np.sqrt(np.mean( (res1-res2)**2 * distr))

def TVD(dens1, dens2) :
    
    grid = np.linspace(0,1,1000).reshape((-1,1))
    
    distr1 = dens1(grid)
    distr1 = distr1/distr1.mean()
    
    distr2 = dens2(grid)
    distr2 = distr2/distr2.mean()
    
    return np.mean( np.abs(distr1-distr2))