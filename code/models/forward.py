#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:01:15 2024

@author: pvillani
"""
from utils.errors import gaussian_error, uniform_error
import utils.analytical_forwards as af

import numpy as np
import math

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
            case 3 :
                self.dout = 14
                # sensors generation
                nlat= 6
                nlon = 4
                i = np.array(list(range(nlat)))
                j = np.array(list(range(nlon)))
                latitude = np.reshape([np.cos(2*i*math.pi/nlat), np.sin(2*i*math.pi/nlat)], (2,1,nlat))
                longitude = np.reshape([np.sin(j*math.pi/(nlon-1)), np.cos(j*math.pi/(nlon-1))], (2, nlon,1))
                
                sensors= np.array( [latitude[0]*longitude[0], latitude[1]*longitude[0], np.ones((1,6)) *longitude[1]], dtype=np.float16)
                self.sensors = np.unique(np.reshape(sensors, (3, -1)).T, axis = 0).astype(float)

                self.forward = self.d3_model
                
            case 4 :
                self.dout = 12
                theta = np.linspace(0, 2*math.pi, self.dout+1)[:self.dout]
                self.sensors = np.transpose( [np.sin(theta), np.cos(theta) ] )
                self.forward = self.d4_model
            case _ :
                raise NotImplementedError("only 1d case as for now")
            
        self.dom = dom
                
        if error == "N" :
            self.discr_err = gaussian_error
        else :
            self.discr_err = uniform_error
        
    def predict(self, p, tols = 0, **kwargs):
        p = np.reshape(p, (-1, self.dim))
        n_pts = len(p)
        
        val = self.forward(p, **kwargs)
        
        val = val.reshape( (n_pts, -1))
        if np.all(tols > 0):
            return self.discr_err(val, tols), tols
        else:
            return val
        
    def d4_model(self, x, dom = None) :
        return af.stationary_heat_eq(x, self.sensors)

    def d3_model(self, x) :
        return af.d3_laplace(x, self.sensors)
        