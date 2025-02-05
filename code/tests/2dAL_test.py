#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:02:01 2024

@author: pvillani
"""
import numpy as np
import matplotlib.pyplot as plt

from models.forward import forward_model as fm
from models.GP_models.MTSurrogate import MTModel
from models.lipschitz import lipschitz_regressor

from IP.likelihoods import base_likelihood, lipschitz_likelihood, GP_likelihood

from AL.exp_err_red import pos_EER
from emcee import EnsembleSampler

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from utils.metrics import experror, TVD

reproducibility_seed(12)

dim = 2

path = "./outputs/AL2d"
import os
if not os.path.exists(path):
    os.makedirs(path)

# measurements and measurements variance
meas = np.array( [[0.5, 0.9]])
eps_l = np.array([0.05, 0.06])

# ground truth
forward = fm(2, "U")
true_like = base_likelihood(meas, eps_l**2,forward)

# initial number of training points
n_init = 3
# final number of training points
n_max = 23

# noise level
noise_level = 0.01

rg = 1

n_test = 100
test = np.linspace(0,1, n_test)
X, Y = np.meshgrid(test, test)
X = X.reshape((-1))
Y = Y.reshape((-1))
x_test = np.transpose([X,Y])
# plot posteriors
fig, axs = plt.subplots(1,3,figsize = (12,3))

truth = true_like.plugin(x_test)
truth /= truth.mean()
axs[0].contourf(test, test, truth.reshape((n_test,n_test)), 40)

fig.savefig(path + "/posterior_comparison.png", format = 'png', transparent = True)

# training points
x_init = []
y_init = []
noise_init = noise_level * np.ones((n_init))
for i in range(rg) :
    x_init.append(lhs( np.zeros(dim), np.ones(dim), n_init).reshape((-1,dim)))
    y_i, _ = forward.predict(x_init[i], noise_init)
    y_init.append(y_i)

# discretized domain
fix = np.linspace(0,1, 100)
X, Y = np.meshgrid(fix, fix)
X = X.reshape((-1))
Y = Y.reshape((-1))
grid = np.transpose([X,Y])

# iterate results 
# EEGP = np.zeros((rg,n_max - n_init + 1 ) )
# L2GP = np.zeros((rg,n_max - n_init + 1 ) )
# for i in range(rg) :
#     ## active learning, GP
#     x = x_init[i]
#     noise = noise_init
#     y = y_init[i]
#     n_pts = n_init
    
#     # Gp model
#     GP = MTModel()
#     GP.fit(x, y, noise = noise**2/3)
#     # likeliood
#     GPlike = GP_likelihood(meas, eps_l**2, GP)
    
#     # EEGP[i, n_pts - n_init] = experror(GP, forward, true_like.plugin )
#     # L2GP[i, n_pts - n_init] = TVD(GPlike.marginal, true_like.plugin)
    
#     while n_pts < n_max :
#         # predict std
#         _, std = GP.predict(grid, return_std = True)
#         # acquisition function
#         # if n_pts %2 == 0 :
#         #     acq_fun = np.exp(1/noise_level * std.reshape(-1)) * GPlike.marginal(grid)
#         # else :
#         #     acq_fun =(std).reshape(-1)
#         acq_fun = np.exp(1/noise_level * np.mean(std, axis = 1 )) * GPlike.marginal(grid)
#         # find maximizer of acquisition
#         nx = np.array( [grid[np.argmax(acq_fun)]])
#         # evaluate model
#         nnoise = noise_level * np.ones((1))
#         ny, _ = forward.predict(nx, nnoise)
#         # update training set
#         x = np.append(x, nx, axis = 0)
#         y = np.append(y, ny, axis = 0)
#         noise = np.append(noise,nnoise, axis = 0)
#         n_pts = n_pts+1
#         # retrain model
#         GP.fit(x, y, noise = noise**2/3)
        
#         # EEGP[i, n_pts - n_init] = experror(GP, forward, true_like.plugin )
#         # L2GP[i, n_pts - n_init] = TVD(GPlike.marginal, true_like.plugin)
    

# GP_x = x
# GP_y = y
# save data
# with open(path + "/tr_set/GP_x.npy", 'wb') as file :
#     np.save(file, x)
# with open(path + "/tr_set/GP_y.npy", 'wb') as file :
#     np.save(file, y)

EELR = np.zeros((rg,n_max - n_init + 1 ) )
L2LR = np.zeros((rg,n_max - n_init + 1 ) )
for i in range(rg) :
    ## active learning, LR
    x = x_init[i]
    noise = noise_init
    y = y_init[i]
    n_pts = n_init
    
    # LR model
    lips = lipschitz_regressor(x, y, noise)
    # likelihood
    liplike = lipschitz_likelihood(meas, eps_l**2, lips)
    
    # EELR[i, n_pts - n_init] = experror(lips, forward, true_like.plugin )
    # L2LR[i, n_pts - n_init] = TVD(liplike.marginal, true_like.plugin)
    sampler = EnsembleSampler(nwalkers = 32, ndim = 2, log_prob_fn= liplike.log_marginal)
    start = np.random.uniform(0,1, size = (32,2))
    
    while n_pts < n_max :
        sampler.run_mcmc(initial_state=start, nsteps = 100)
        start = sampler.get_last_sample()
        sampler.reset()
        sampler.run_mcmc(start, 400)
        samples = sampler.get_chain(flat = True)
        start = sampler.get_last_sample()
        sampler.reset()
        
        acq_fun = np.zeros( len(grid))

        for i,p in enumerate(grid) :
            p = p.reshape((1,2))

            acq_fun[i] = pos_EER(noise_level, p, samples, lips).mean()


        # find maximizer of acquisition
        nx = np.array( [grid[np.argmax(acq_fun)]])
        # evaluate model
        nnoise = noise_level * np.ones((1))
        ny, _ = forward.predict(nx, nnoise)
        # update training set
        x = np.append(x, nx, axis = 0)
        y = np.append(y, ny, axis = 0)
        noise = np.append(noise,nnoise, axis = 0)
        n_pts = n_pts+1
        #print(lips.L)
        # retrain model
        lips.update(nx, ny, nnoise)
        
        plt.figure()
        plt.contourf(fix, fix, acq_fun.reshape((100,100)) ) 
        plt.scatter(x[:-1, 0], x[:-1, 1])
        plt.title(f'iteration {n_pts - n_init} ')
        plt.savefig(path+f"/acq/it{n_pts - n_init}.png", format = 'png')
        plt.close()
        
        fig, axs = plt.subplots(1,3,figsize = (12,3))
        lip_marginal = liplike.marginal(x_test)
        lip_marginal /= lip_marginal.mean()
        axs[1].contourf(test, test, lip_marginal.reshape((n_test,n_test)),40) 
        lip_plug = liplike.plugin(x_test)
        lip_plug /= lip_plug.mean()
        axs[2].contourf(test, test, lip_plug.reshape((n_test,n_test)),40)
        # GP_marginal = GPlike.marginal(x_test)
        # GP_marginal /= GP_marginal.mean()
        # axs[2].contourf(test, test, GP_marginal.reshape((n_test,n_test)),40)
        truth = true_like.plugin(x_test)
        truth /= truth.mean()
        axs[0].contourf(test, test, truth.reshape((n_test,n_test)), 40)

        fig.savefig(path + "/posterior_comparison.png", format = 'png', transparent = True)
        plt.close('all')

        x_ran = list(range(n_init, n_max+1))
        # EELR[i, n_pts - n_init] = experror(lips, forward, true_like.plugin )
        # L2LR[i, n_pts - n_init] = TVD(liplike.marginal, true_like.plugin)
    
# # save data
# with open(path + "/tr_set/LR_x.npy", 'wb') as file :
#     np.save(file, x)
# with open(path + "/tr_set/LR_y.npy", 'wb') as file :
#     np.save(file, y)
    
n_test = 100
test = np.linspace(0,1, n_test)
X, Y = np.meshgrid(test, test)
X = X.reshape((-1))
Y = Y.reshape((-1))
x_test = np.transpose([X,Y])

# plot models
fig, axs = plt.subplots(2,3,figsize = (12,6))

pred = forward.predict(x_test).reshape( (n_test,n_test,-1))
lippred = lips.predict(x_test).reshape( (n_test,n_test,-1))
#GPpred = GP.predict(x_test).reshape( (n_test,n_test,-1))
for i in range(2):
    axs[i,0].contourf(test, test, pred[:,:,i], 40 )
    axs[i,1].contourf(test, test, lippred[:,:,i], 40 )
    # axs[i,2].contourf(test, test, GPpred[:,:,i], 40 )

    
fig.savefig(path + "/model_comparison.png", format = 'png', transparent = True)

# plot posteriors
fig, axs = plt.subplots(1,3,figsize = (12,3))
lip_marginal = liplike.marginal(x_test)
lip_marginal /= lip_marginal.mean()
axs[1].contourf(test, test, lip_marginal.reshape((n_test,n_test)),40) 
lip_plug = liplike.plugin(x_test)
lip_plug /= lip_plug.mean()
axs[2].contourf(test, test, lip_plug.reshape((n_test,n_test)),40)
# GP_marginal = GPlike.marginal(x_test)
# GP_marginal /= GP_marginal.mean()
# axs[2].contourf(test, test, GP_marginal.reshape((n_test,n_test)),40)
truth = true_like.plugin(x_test)
truth /= truth.mean()
axs[0].contourf(test, test, truth.reshape((n_test,n_test)), 40)

fig.savefig(path + "/posterior_comparison.png", format = 'png', transparent = True)

x_ran = list(range(n_init, n_max+1))


# fig, ax = plt.subplots(figsize = (12,3))
# for i in range(rg):
#     ax.semilogy(  x_ran, L2GP[i], alpha = 0.2, color = 'blue')
#     ax.semilogy( x_ran , L2LR[i], alpha = 0.2, color = 'red')
# ax.semilogy(  x_ran, np.mean(L2GP, axis = 0 ), label = "GPR posterior", color = 'blue')
# ax.semilogy( x_ran , np.mean(L2LR, axis = 0 ), label = "LR posterior", color = 'red')
# plt.title("Total variation distance between posteriors")
# plt.legend()
# plt.xticks(x_ran)
# fig.savefig(path + "/TV_convergence.png", format = 'png', transparent = True)

# fig, ax = plt.subplots(figsize = (12,3))
# for i in range(rg):
#     ax.semilogy(  x_ran, EEGP[i], alpha = 0.2, color = 'blue')
#     ax.semilogy( x_ran , EELR[i], alpha = 0.2, color = 'red')
# ax.semilogy(  x_ran, np.mean(EEGP, axis = 0 ), label = "GPR model", color = 'blue')
# ax.semilogy( x_ran , np.mean(EELR, axis = 0 ), label = "LR model", color = 'red')
# plt.title("Expected error")
# plt.legend()
# plt.xticks(x_ran)
# fig.savefig(path + "/EE_convergence.png", format = 'png', transparent = True)
