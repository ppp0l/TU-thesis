#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:49:09 2024

@author: pvillani
"""
import numpy as np

def gaussian_error(mean, std) :
    if len(np.shape(std) ) ==1 :
        std = np.array( [ std[i]*np.eye(len(mean[0])) for i in range(len(mean)) ] )
    elif len(np.shape(std) ) ==2 :
        std = np.reshape(std, (len(std), -1))

    normal = np.random.normal(size = np.shape(mean))
    normal = normal.reshape((len(normal), -1, 1))

    noise = np.reshape(std @ normal, mean.shape)
    return mean + noise

def uniform_error(mean, bound) :
    if len(np.shape(bound) ) <2 :
        bound = np.outer(bound, np.ones(len(mean[0])))
    return np.random.uniform( mean - bound, mean + bound)