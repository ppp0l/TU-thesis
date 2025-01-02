#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:12:07 2024

@author: pvillani
"""
import numpy as np
import matplotlib.pyplot as plt

from models.forward import forward_model as fm
from models.GP_models.SOSurrogate import SOGPModel
from models.lipschitz import lipschitz_regressor

from likelihoods.likelihoods import lipschitz_likelihood, GP_likelihood

from utils.utils import latin_hypercube_sampling as lhs, reproducibility_seed

reproducibility_seed(42)

path = "./outputs"
import os
if not os.path.exists(path):
    os.makedirs(path)

# parameter space
param_space = { 
    'min' : np.array([0.0]), 
    'max' : np.array([2.0])
}

# training points, x
x = lhs( 0, 1, 15)

# ground truth
forward = fm(1, "U")
true = forward.predict(x)

# training points
noise_level = 0.1 * np.ones_like(true)# np.abs(np.random.normal(0, 0.2, size = true.shape) )
y = np.random.uniform(true-noise_level,true+ noise_level)
#train_y = np.random.normal(loc = true, scale = noise_level/2 )

# Gp model
GP = SOGPModel()
GP.fit(x, y, noise = noise_level**2/4)

# LR model
lips = lipschitz_regressor(x, y, noise_level)

# testevaluation
test = np.linspace(0,1, 1000)
x_test = test.reshape((-1,1) )
mean, std = GP.predict(x_test, return_std = True)
pred, low_bd, up_bd = lips.predict(test, return_bds=True)

# measurements and measurements likelihood
meas = np.array( [[0.5]])
eps_l = np.array([0.05])

# plot models
fig, ax = plt.subplots(figsize = (12,3))
ax.plot(test, forward.predict(test), 'black', lw = 1, label = "Ground truth", zorder = 8)

ax.scatter(x, y, color = 'black', marker = '*', label = 'Training points', zorder = 10)
#ax.scatter(x, train_y, color = 'red', marker = '*', label = 'GP training points', zorder = 10)


ax.plot(test, mean, 'blue', lw = 1, label="GP predictive mean and \n 95% confidence interval")
ax.fill_between(
    test.ravel(),
    np.reshape(mean - 1.96 * std, -1),
    np.reshape(mean + 1.96 * std, -1),
    alpha=0.2,
    label="GP predictive mean and \n 95% confidence interval",
    facecolor= 'blue'
)

ax.plot(test, pred, 'cyan', lw = 1, label ="Lipschitz prediction and \n confidence interval")
ax.fill_between(
    test.ravel(),
    low_bd[:,0],
    up_bd[:,0],
    alpha=0.2,
    label="Lipschitz prediction and \n confidence interval",
    facecolor= 'cyan'
)

plt.axhline(meas)
plt.axhline(meas + 2 * eps_l, linestyle="dashdot")
plt.axhline(meas - 2 * eps_l, linestyle="dashdot")
handler, labeler = ax.get_legend_handles_labels()
hd = [(handler[0],), (handler[1],),
      (handler[2],handler[3]), 
      (handler[4],handler[5]),
     ]
lab = ["Ground truth", 'Training points',"GP predictive mean and \n 95% confidence interval", "Lipschitz prediction and \n confidence interval", ]
ax.legend(hd, lab, )
#loc = 'upper right', prop={'size': 16} )
fig.savefig(path + "/model_comparison.svg", format = 'svg', transparent = True)

# posteriors
liplike = lipschitz_likelihood(meas, eps_l**2, lips)
GPlike = GP_likelihood(meas, eps_l**2, GP)

# plot posteriors
fig, ax = plt.subplots(figsize = (12,3))
# lip_plugin = liplike.plugin(test)
# lip_plugin /= lip_plugin.mean()
# ax.plot(test, lip_plugin, lw = 1, label = "Lipschitz plugin", color = 'orange')
lip_marginal = liplike.marginal(test)
lip_marginal /= lip_marginal.mean()
ax.plot(test, lip_marginal, lw = 1, label = "Lipschitz marginal", color = 'red')
# GP_plugin = GPlike.plugin(x_test)
# GP_plugin /= GP_plugin.mean()
# ax.plot(test, GP_plugin, lw = 1, label = "GP plugin", color = 'green')
GP_marginal = GPlike.marginal(x_test)
GP_marginal /= GP_marginal.mean()
ax.plot(test, GP_marginal, lw = 1, label = "GP marginal", color = 'blue')
GPlike.model = forward
truth = GPlike.plugin(test)
truth /= truth.mean()
ax.plot(test, truth, lw = 1, label = "Ground truth", color = 'black')

plt.legend()
fig.savefig(path + "/marg_posterior_comparison.svg", format = 'svg', transparent = True)