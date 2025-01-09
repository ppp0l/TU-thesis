#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:45:56 2024

@author: pvillani
"""
import numpy as np
from naive_lipschitz import lipschitz_regressor

class lipschitz_split_regressor(lipschitz_regressor) :
    def __init__(self, train_x, train_y, noise):
        self.train_x = train_x.reshape( (len(train_x), -1))
        self.train_y = train_y.reshape( (len(train_y), -1))
        self.noise = noise

        self.dim = len(self.train_x[0])
        self.dout= len(self.train_y[0])

        self.L = 0
        self.splits= []
        self.split_training()
        self.estimate_constant()
        self.find_split()
        
        

    def predict(self,x, return_bds = False) :
        
        #TODO: only works 1d

        tr_y = self.train_y.reshape((1, -1, self.dout)  )
        tr_x = self.train_x
        noise = self.noise.reshape((1, -1, 1))
        x = x.reshape( (-1, self.dim))

        L = self.L

        x2 = np.sum(x**2, axis = 1, keepdims=True)
        tr_x2 = np.sum(tr_x**2, axis = 1, keepdims=True) 
        xxT = x.dot(tr_x.transpose())
        dist_x = np.sqrt( x2 + tr_x2.transpose() - 2 *xxT )
        shape = dist_x.shape
        Ldist = np.outer(dist_x.reshape(-1 ), L).reshape( (shape[0],shape[1],len(L)))


        low_bd = np.max( tr_y - noise  - Ldist , axis = 1)
        up_bd = np.min( tr_y + noise  + Ldist , axis = 1)

        pred = (low_bd + up_bd)/2
        
        if return_bds :

            return pred, low_bd, up_bd
        
        return pred
    
    def split_training(self) :
        
        xs = [self.train_x]
        ys = [self.train_y]

        for split in self.splits:
            pt = split.p
            w = split.w
            for i,x in enumerate(xs) :
                if pt in x :
                    above = np.dot(x, w) >= np.dot(pt, w)
                    x1 = x[ above ]
                    x2 = x[ ~above ]
                    y1 = ys[i][ above ]
                    y2 = ys[i][ ~above ]
                    break
            xs.remove(x)
            xs.append(x1)
            xs.append(x2)
            ys.remove(ys[i])
            ys.append(y1)
            ys.append(y2)
            
            
        self.splitted_x = xs
        self.splitted_y = ys
    
    def add_split(self, split) :
        xs = self.splitted_x
        ys = self.splitted_y

        pt = split.p
        w = split.w
        for i,x in enumerate(xs) :
            if pt in x :
                above = np.dot(x, w) >= np.dot(pt, w)
                x1 = x[ above ]
                x2 = x[ ~above ]
                y1 = ys[i][ above ]
                y2 = ys[i][ ~above ]
                break
        xs.remove(x)
        xs.append(x1)
        xs.append(x2)
        ys.remove(ys[i])
        ys.append(y1)
        ys.append(y2)

        self.splitted_x = xs
        self.splitted_y = ys
    
    def find_split(self)  :

        x = self.train_x

        curr_splits = self.splits


    
    def estimate_slice_constant(self, x, y) :
        
        x2 = np.sum(x**2, axis = 1, keepdims=True)
        xxT = x.dot(x.transpose())
        dist_x = np.sqrt( x2 + x2.transpose() - 2 *xxT )
        
        y = y.transpose()
        y = y.reshape( (len(y), len(y[0]), -1 ))
        
        dist_y = np.abs(y - y.transpose((0,2,1)))
        
        eps = self.noise.reshape((len(y[0]), -1))
        eps2 = (eps + eps.transpose())
        
        L_mat = (dist_y )/dist_x - eps2/dist_x
        
        L = np.nanmax(L_mat, axis = (1,2) )*1.1
        
        return L
        #self.L = 3
        
    def update(self, nx, ny, nnoise) :
        
        self.train_x = np.concatenate((self.train_x,nx))
        self.train_y = np.concatenate((self.train_y,ny))
        self.noise = np.concatenate((self.noise,nnoise))
        
        self.split_training()
        self.estimate_constant()
        self.find_split()
