#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:01:15 2024

@author: pvillani
"""
from utils.errors import gaussian_error, uniform_error
import utils.analytical_forwards as af

import numpy as np

class forward_model() :
    def __init__(self, dim, error = "N", dom = None) :
        
        self.dim = dim
        match dim :
            case 1 :
                self.forward = af.d1_model
                self.dout = 1
            case 2 :
                self.forward = af.d2_model
                self.dout = 2
            case _ :
                raise NotImplementedError("only 1d case as for now")
            
        self.dom = dom
                
        if error == "N" :
            self.discr_err = gaussian_error
        else :
            self.discr_err = uniform_error
        
    def predict(self, p, tols = 0, **kwargs):
         
        n_pts = len(p)
        
        val = self.forward(p, **kwargs)
        
        val = val.reshape( (n_pts, -1))
        if np.all(tols > 0):
            return self.discr_err(val, tols), tols
        else:
            return val