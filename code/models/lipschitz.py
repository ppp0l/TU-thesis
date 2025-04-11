#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:45:56 2024

@author: pvillani
"""
import numpy as np
from models.surrogate import Surrogate

class lipschitz_regressor(Surrogate) :
    def __init__(self, dim, dout):

        self.dim = dim
        self.dout= dout

        self.L = 0

    def fit(self, train_x, train_y, noise = None):
        
        if noise is None:
            noise = np.zeros_like(train_y)
        elif noise.ndim == 1:
            noise = noise.reshape(-1, 1) * np.ones((1, self.dout))

        self.train_x = train_x.reshape( (-1, self.dim))
        self.train_y = train_y.reshape( (-1, self.dout))
        self.noise = noise.reshape( (-1, self.dout))
        
        self.estimate_constant(self.train_x, self.train_y, self.noise)
        
        

    def predict(self,x, return_bds = False, return_dp = False) :

        tr_y = self.train_y.reshape((1, -1, self.dout)  )
        tr_x = self.train_x
        noise = self.noise.reshape((1, -1, self.dout))
        x = np.reshape( x, (-1, self.dim))

        L = self.L

        x2 = np.sum(x**2, axis = 1, keepdims=True)
        tr_x2 = np.sum(tr_x**2, axis = 1, keepdims=True) 
        xxT = x.dot(tr_x.transpose())
        dist_sq = x2 + tr_x2.transpose() - 2 *xxT
        dist_sq[dist_sq <0] = 0
        dist_x = np.sqrt( dist_sq )
        Ldist = np.outer(dist_x.reshape(-1 ), L).reshape( (len(x),len(tr_x),len(L)))
        
        low_bd = np.max( tr_y - noise  - Ldist , axis = 1)
        up_bd = np.min( tr_y + noise  + Ldist , axis = 1)

        
        pred = (low_bd + up_bd)/2

        if return_dp :
            # non vectorialized!! only gets called for single x so ok
            dlb = np.zeros( (self.dim, 1, self.dout))
            dub = np.zeros( (self.dim, 1, self.dout))

            arg_low = np.argmax(tr_y - noise  - Ldist, axis = 1)[0]
            low_x = tr_x[arg_low]
            ddist = - (x-low_x) / np.linalg.norm( x - low_x, ord = 2, axis=1, keepdims=True)
            dlb[:,0,:] = np.transpose(ddist).reshape((self.dim, -1) ) * L

            arg_up = np.argmin(tr_y + noise  + Ldist, axis = 1)[0]
            up_x = tr_x[arg_up]
            ddist = - (x-up_x) / np.linalg.norm( x - up_x, ord = 2, axis=1, keepdims=True)
            dub[:,0,:] = np.transpose(ddist) * L

            if return_bds :
                return pred, low_bd, up_bd,  dlb, dub

            dpred = (dlb + dub) / 2

            return pred, dpred
        
        if return_bds :

            return pred, low_bd, up_bd
        
        return pred

    
    def estimate_constant(self, x,y, noise) :
        
        x2 = np.sum(x**2, axis = 1, keepdims=True)
        xxT = x.dot(x.transpose())
        dist_sq = x2 + x2.transpose() - 2 *xxT
        dist_sq[dist_sq <0] = 0
        dist_x = np.sqrt( dist_sq )
        n_pts = len(y)
        y = y.transpose()
        y = y.reshape( (self.dout, n_pts, -1 ))
        
        dist_y = np.abs(y - y.transpose((0,2,1)))
        
        eps = noise.reshape( (self.dout, n_pts, -1 ))
        eps2 = (eps + eps.transpose((0,2,1)))
        
        L_mat = (dist_y )/dist_x - eps2/dist_x

        alpha = 1.1 + np.exp(- len(x)/(self.dim**2))
        
        L = np.nanmax(L_mat, axis = (1,2) )*alpha
        
        self.L = L
        #self.L = 3
        
    def update(self, nx, ny, nnoise = None, updated_y = None, updated_noise = None) :
        if nnoise is None:
            nnoise = np.zeros_like(ny)
            
        nx = nx.reshape( (-1, self.dim))
        ny = ny.reshape( (-1, self.dout))
        nnoise = nnoise.reshape( (-1, self.dout))

        if updated_y is not None:
            train_y = updated_y.reshape( (-1, self.dout))
        else:
            train_y = self.train_y

        if updated_noise is not None:
            noise = updated_noise.reshape( (-1, self.dout))
        else:
            noise = self.noise        
        
        self.train_x = np.concatenate((self.train_x,nx))
        self.train_y = np.concatenate((train_y,ny))
        self.noise = np.concatenate((noise,nnoise))
        
        self.estimate_constant(self.train_x, self.train_y, self.noise)

    def state_dict(self):
        return {'L': self.L}

    def load_state_dict(self, state_dict):
        self.L = state_dict['L']


    def predict_acc(self,
                    Ldist :np.ndarray, 
                    y :np.ndarray, 
                    accs :np.ndarray,
                    return_deps : bool = False
                    ) :
        n_tr = len(accs)
        n_pred = len(Ldist)
        
        accs = accs.reshape((n_tr, -1))
        accs = accs * np.ones( (1, self.dout))
        
        noise = accs.reshape((1, -1, self.dout))

        low_bd = np.max( y - noise  - Ldist , axis = 1)
        up_bd = np.min( y + noise  + Ldist , axis = 1)

        if return_deps :
            dlb = np.zeros( (n_tr, n_pred, self.dout))
            dub = np.zeros( (n_tr, n_pred, self.dout))

            arg_low = np.argmax(y - noise  - Ldist, axis = 1)

            arg_up = np.argmin(y + noise  + Ldist, axis = 1)

            # there's surely a better way to do this, but ok
            for i in range(n_pred) :
                for j in range(self.dout) :
                    dlb[arg_low[i,j], i, j] = - 1
                    dub[arg_up[i,j], i, j ] = 1

            return low_bd, up_bd, dlb,  dub
        
        return low_bd, up_bd
    
    def compute_Ldist(self, pts, oth_tr = None) :
        tr_x = self.train_x
        if oth_tr is None :
            all_tr = tr_x
        else :
            all_tr = np.concatenate((tr_x, oth_tr)) 
        
        
        tr_x2 = np.sum(tr_x**2, axis = 1, keepdims=True)
        trtrT = tr_x.dot(tr_x.transpose())
        dist_sq = tr_x2 + tr_x2.transpose() - 2 *trtrT
        dist_sq[dist_sq <0] = 0
        dist_x = np.sqrt( dist_sq )

        y = self.train_y.transpose()
        y = y.reshape( (self.dout, len(self.train_y), -1 ))
        
        dist_y = np.abs(y - y.transpose((0,2,1)))
        
        L_mat = (dist_y )/dist_x 
        L = np.nanmax(L_mat, axis = (1,2) )#*1.2

        all_tr2 = np.sum(all_tr**2, axis = 1, keepdims=True)
        pts2 = np.sum(pts**2, axis = 1, keepdims=True)
        xxT = pts.dot(all_tr.transpose())
        dist_sq = pts2 + all_tr2.transpose() - 2 *xxT
        dist_sq[dist_sq <0] = 0
        dist_pt_tr = np.sqrt( dist_sq)
        Ldist = np.outer(dist_pt_tr.reshape(-1 ), L).reshape( (len(pts),len(all_tr),len(L)))

        return Ldist