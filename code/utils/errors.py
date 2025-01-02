#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:49:09 2024

@author: pvillani
"""
import numpy as np

def gaussian_error(mean, std) :
    if len(np.shape(std) ) <2 :
        std = np.outer(std, np.ones(len(mean[0])))
    return np.random.normal(loc = mean, scale = std/np.sqrt(3) )

def uniform_error(mean, bound) :
    if len(np.shape(bound) ) <2 :
        bound = np.outer(bound, np.ones(len(mean[0])))
    return np.random.uniform( mean - bound, mean + bound)