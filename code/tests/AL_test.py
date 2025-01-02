#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:02:01 2024

@author: pvillani
"""
import numpy as np
import matplotlib.pyplot as plt

from models.forward import forward_model as fm
from models.GP_models.SOSurrogate import SOGPModel
from models.lipschitz import lipschitz_regressor

from likelihoods.likelihoods import base_likelihood, lipschitz_likelihood, GP_likelihood

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed
from utils.metrics import experror, TVD

reproducibility_seed(22)

path = "./outputs/AL"
import os
if not os.path.exists(path):
    os.makedirs(path)

# measurements and measurements variance
meas = np.array( [[0.3]])
eps_l = np.array([0.1])

# ground truth
forward = fm(1, "N")
true_like = base_likelihood(meas, eps_l**2,forward)

# initial number of training points
n_init = 3
# final number of training points
n_max = 13

# noise level
noise_level = 0.05

rg = 10

# training points
x_init = []
y_init = []
noise_init = noise_level * np.ones((n_init, 1))
for i in range(rg) :
    x_init.append(lhs( 0, 1, n_init).reshape((-1,1)))
    y_i, _ = forward.predict(x_init[i], noise_init)
    y_init.append(y_i)

# discretized domain
grid = np.linspace(0,1, 200).reshape((-1,1))

# iterate results 
EEGP = np.zeros((rg,n_max - n_init + 1 ) )
L2GP = np.zeros((rg,n_max - n_init + 1 ) )
for i in range(rg) :
    ## active learning, GP
    x = x_init[i]
    noise = noise_init
    y = y_init[i]
    n_pts = n_init
    
    # Gp model
    GP = SOGPModel()
    GP.fit(x, y, noise = noise**2/3)
    # likeliood
    GPlike = GP_likelihood(meas, eps_l**2, GP)
    
    EEGP[i, n_pts - n_init] = experror(GP, forward, true_like.plugin )
    L2GP[i, n_pts - n_init] = TVD(GPlike.marginal, true_like.plugin)
    
    while n_pts < n_max :
        # predict std
        _, std = GP.predict(grid, return_std = True)
        # acquisition function
        # if n_pts %2 == 0 :
        #     acq_fun = np.exp(1/noise_level * std.reshape(-1)) * GPlike.marginal(grid)
        # else :
        #     acq_fun =(std).reshape(-1)
        acq_fun = np.exp(1/noise_level * std.reshape(-1)) * GPlike.marginal(grid)
        # find maximizer of acquisition
        nx = np.array( [grid[np.argmax(acq_fun)]])
        # evaluate model
        nnoise = noise_level * np.ones((1, 1))
        ny, _ = forward.predict(nx, nnoise)
        # update training set
        x = np.append(x, nx, axis = 0)
        y = np.append(y, ny, axis = 0)
        noise = np.append(noise,nnoise, axis = 0)
        n_pts = n_pts+1
        # retrain model
        GP.fit(x, y, noise = noise**2/3)
        
        EEGP[i, n_pts - n_init] = experror(GP, forward, true_like.plugin )
        L2GP[i, n_pts - n_init] = TVD(GPlike.marginal, true_like.plugin)
    

GP_x = x
GP_y = y
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
    
    EELR[i, n_pts - n_init] = experror(lips, forward, true_like.plugin )
    L2LR[i, n_pts - n_init] = TVD(liplike.marginal, true_like.plugin)
    
    while n_pts < n_max :
        # predict std
        _, low_bd, up_bd = lips.predict(grid, return_bds=True)
        # acquisition function
        if n_pts %3 > -1 :
            acq_fun = np.exp(2/noise_level *(up_bd-low_bd)).reshape(-1) * liplike.marginal(grid)
        else :
            acq_fun =(up_bd-low_bd).reshape(-1)
        # find maximizer of acquisition
        nx = np.array( [grid[np.argmax(acq_fun)]])
        # evaluate model
        nnoise = noise_level * np.ones((1, 1))
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
        plt.plot(grid, acq_fun)
        plt.scatter(x[:-1],np.zeros_like(x[:-1]))
        plt.title(f'iteration {n_pts - n_init} ')
        plt.savefig(path+f"/acq/it{n_pts - n_init}.png", format = 'png')
        plt.close()
        
        EELR[i, n_pts - n_init] = experror(lips, forward, true_like.plugin )
        L2LR[i, n_pts - n_init] = TVD(liplike.marginal, true_like.plugin)
    
# # save data
# with open(path + "/tr_set/LR_x.npy", 'wb') as file :
#     np.save(file, x)
# with open(path + "/tr_set/LR_y.npy", 'wb') as file :
#     np.save(file, y)
    

test = np.linspace(0,1, 1000).reshape((-1,1))

# plot models
fig, ax = plt.subplots(figsize = (12,3))
ax.plot(test, forward.predict(test), 'black', lw = 1, label = "Ground truth", zorder = 8)

ax.scatter(x, y, color = 'black', marker = '*', label = 'Training points', zorder = 10)


mean, std = GP.predict(test, return_std = True)

ax.plot(test, mean, 'blue', lw = 1, label="GP predictive mean and \n 95% confidence interval")
ax.fill_between(
    test.ravel(),
    np.reshape(mean - 1.96 * std, -1),
    np.reshape(mean + 1.96 * std, -1),
    alpha=0.2,
    label="GP predictive mean and \n 95% confidence interval",
    facecolor= 'blue'
)

pred, low_bd, up_bd = lips.predict(test, return_bds=True)
ax.plot(test, pred, 'cyan', lw = 1, label ="Lipschitz prediction and \n confidence interval")
ax.fill_between(
    test.ravel(),
    low_bd[:,0],
    up_bd[:,0],
    alpha=0.2,
    label="Lipschitz prediction and \n confidence interval",
    facecolor= 'cyan'
)
ax.scatter(GP_x, GP_y, color = 'red', marker = '*', label = 'GP training points', zorder = 9)
plt.axhline(meas)
plt.axhline(meas + 2 * eps_l, linestyle="dashdot")
plt.axhline(meas - 2 * eps_l, linestyle="dashdot")

handler, labeler = ax.get_legend_handles_labels()
hd = [(handler[0],), (handler[1],),
      (handler[6],),
      (handler[2],handler[3]), 
      (handler[4],handler[5]),
     ]

lab = ["Ground truth", 'Lips training points','GP training points', "GP predictive mean and \n 95% confidence interval", "Lipschitz prediction and \n confidence interval", ]
ax.legend(hd, lab, )
#loc = 'upper right', prop={'size': 16} )
fig.savefig(path + "/model_comparison.svg", format = 'svg', transparent = True)


# plot posteriors
fig, ax = plt.subplots(figsize = (12,3))
lip_marginal = liplike.marginal(test)
lip_marginal /= lip_marginal.mean()
ax.plot(test, lip_marginal, lw = 1, label = "Lipschitz marginal", color = 'red')
GP_marginal = GPlike.marginal(test)
GP_marginal /= GP_marginal.mean()
ax.plot(test, GP_marginal, lw = 1, label = "GP marginal", color = 'blue')
truth = true_like.plugin(test)
truth /= truth.mean()
ax.plot(test, truth, lw = 1, label = "Ground truth", color = 'black')

plt.legend()
fig.savefig(path + "/posterior_comparison.svg", format = 'svg', transparent = True)

x_ran = list(range(n_init, n_max+1))


fig, ax = plt.subplots(figsize = (12,3))
for i in range(rg):
    ax.semilogy(  x_ran, L2GP[i], alpha = 0.2, color = 'blue')
    ax.semilogy( x_ran , L2LR[i], alpha = 0.2, color = 'red')
ax.semilogy(  x_ran, np.mean(L2GP, axis = 0 ), label = "GPR posterior", color = 'blue')
ax.semilogy( x_ran , np.mean(L2LR, axis = 0 ), label = "LR posterior", color = 'red')
plt.title("Total variation distance between posteriors")
plt.legend()
plt.xticks(x_ran)
fig.savefig(path + "/TV_convergence.svg", format = 'svg', transparent = True)

fig, ax = plt.subplots(figsize = (12,3))
for i in range(rg):
    ax.semilogy(  x_ran, EEGP[i], alpha = 0.2, color = 'blue')
    ax.semilogy( x_ran , EELR[i], alpha = 0.2, color = 'red')
ax.semilogy(  x_ran, np.mean(EEGP, axis = 0 ), label = "GPR model", color = 'blue')
ax.semilogy( x_ran , np.mean(EELR, axis = 0 ), label = "LR model", color = 'red')
plt.title("Expected error")
plt.legend()
plt.xticks(x_ran)
fig.savefig(path + "/EE_convergence.svg", format = 'svg', transparent = True)
