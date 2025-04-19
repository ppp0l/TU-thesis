#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:01:15 2024

@author: pvillani
"""
from utils.errors import gaussian_error, uniform_error
from utils.utils import discretization_error_matrix
import utils.analytical_forwards as af

import numpy as np
import math

class forward_model() :
    def __init__(self, dim, error = "N", dom = None, corr = False) :
        
        self.dim = dim
        match dim :
            case 1 :
                self.forward = af.d1_model
                self.dout = 1
                self.eval_std = np.eye(self.dout)
            case 2 :
                self.forward = af.d2_model
                self.dout = 2
                self.eval_std = np.eye(self.dout)
            case 3 :
                self.dout = 14
                # sensors generation
                nlat= 6
                nlon = 4
                i = np.array(list(range(nlat)))
                j = np.array(list(range(nlon)))
                latitude = np.reshape([np.cos(2*i*math.pi/nlat), np.sin(2*i*math.pi/nlat)], (2,1,nlat))
                longitude = np.reshape([np.sin(j*math.pi/(nlon-1)), np.cos(j*math.pi/(nlon-1))], (2, nlon,1))
                
                sensors= np.array( [latitude[0]*longitude[0], latitude[1]*longitude[0], np.ones((1,nlat)) *longitude[1]], dtype=np.float16)
                self.sensors = np.unique(np.reshape(sensors, (3, -1)).T, axis = 0).astype(float)

                self.forward = self.d3_model

                if corr and error == "N" :
                    self.eval_std = discretization_error_matrix(self.sensors)
                else :
                    self.eval_std = np.eye(self.dout)
                
            case 4 :
                self.dout = 12
                theta = np.linspace(0, 2*math.pi, self.dout+1)[:self.dout]
                self.sensors = np.transpose( [np.sin(theta), np.cos(theta) ] )
                self.forward = self.d4_model

            case 6 :
                self.dout = 30

                # sensors generation
                nlat= 3
                nlon = 5
                i = np.array(list(range(nlat)))
                j = np.array(list(range(nlon)))
                latitude = np.reshape([np.cos(2*i*math.pi/nlat), np.sin(2*i*math.pi/nlat)], (2,1,nlat))
                longitude = np.reshape([np.sin(j*math.pi/(nlon-1)), np.cos(j*math.pi/(nlon-1))], (2, nlon,1))
                
                sensors= np.array( [latitude[0]*longitude[0], latitude[1]*longitude[0], np.ones((1,nlat)) *longitude[1]])
                sensors = np.reshape(sensors, (3, -1)).T  
                
                n_eq = 7
                i = np.array(list(range(n_eq)))
                equator =np.array( [np.cos(2*i*math.pi/n_eq), np.sin(2*i*math.pi/n_eq),  np.zeros(n_eq)]).T

                n_up = 0
                n_down = 0
                n_mid = 0
                skipped_u = False
                skipped_d = False
                for i, sensor in enumerate(sensors) :
                    if sensor[2] > 0.9 :
                        if not skipped_u :
                            skipped_u = True
                            continue
                        sensors[i] = equator[n_up+n_down+n_mid]
                        n_up += 1
                        continue
                    if sensor[2] < -0.9 :
                        if not skipped_d :
                            skipped_d = True
                            continue
                        sensors[i] = equator[n_up+n_down+n_mid]
                        n_down += 1
                        continue
                    if np.abs(sensor[2]) < 0.1 :
                        sensors[i] = equator[n_up+n_down+n_mid]
                        n_mid += 1

                times = [0.3, 0.5]

                self.sensors = np.concatenate( [np.append(sensors, t *np.ones((len(sensors),1)), axis = 1) for t in times], axis = 0)         
                self.forward = self.d6_model  

                if corr and error == "N" :
                    self.eval_std = discretization_error_matrix(self.sensors)
                else :
                    self.eval_std = np.eye(self.dout)
                         

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
            noise_std = tols.reshape((n_pts, 1,1))
            noise_std = np.kron(noise_std, np.eye(self.dout))
            noise_std = noise_std @ self.eval_std 

            resp = self.discr_err(val, noise_std)

            self.true_err = val-resp
            self.last_tols = tols

            return resp, tols
        else:
            return val
        
    def get_residuals(self):
        """
        Simulates residuals in FE evaluations for testing the covariance estimation in the GP model.
        """
        n_pts = len(self.last_tols)

        residuals = []

        tolerances = []
        
        for i in range(n_pts):
            curr_tol = self.last_tols[i]

            tol_levels = -int(np.floor(np.log(curr_tol)))
            if tol_levels < 1 :
                tol_levels = 1
            elif tol_levels > 10 :
                tol_levels = 10
            tols = np.exp( - np.array( list(range(tol_levels)) ) -1 )
            tols = tols * curr_tol / np.exp(-tol_levels)

            tolerances.append(tols)

            true_err = self.true_err[i]

            res = np.random.normal(size = (tol_levels, self.dout, 1))
            res = self.eval_std @ res
            res = res.reshape((tol_levels, -1))
            res = res * tols.reshape((-1,1))

            weights = np.linspace(0.2, 1.5, tol_levels).reshape(-1,1)
            res = (2-weights) * res + weights * true_err.reshape((1, -1))
            res /= np.sqrt( (2-weights)**2 + weights**2 )

            residuals.append(res)

        return residuals, tolerances




    def d4_model(self, x, dom = None) :
        return af.stationary_heat_eq(x, self.sensors)

    def d3_model(self, x) :
        return af.d3_laplace(x, self.sensors)
    
    def d6_model(self, x) :
        return af.d6_diffusion(x, self.sensors)
        