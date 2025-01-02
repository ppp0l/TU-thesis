#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:49:09 2024

@author: pvillani
"""
import numpy as np

def gaussian_error(mean, std) :
    return np.random.normal(loc = mean, scale = std/np.sqrt(3) )

def uniform_error(mean, bound) :
    return np.random.uniform( mean - bound, mean + bound)