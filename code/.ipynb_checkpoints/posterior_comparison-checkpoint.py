#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:21:50 2024

@author: pvillani
"""
import numpy as np
import matplotlib.pyplot as plt
import math, torch, gpytorch
import os

path = "./Outputs/plugin_comparison/"
if not os.path.exists(path):
    os.makedirs(path)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self,) :
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        for i in range(200):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, 200, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ), 
                  end="\r",
                  flush = True)
            
            optimizer.step()

    def predict(self, theta):
        self.eval()
        self.likelihood.eval()
        # Make prediction
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            prediction = self(theta)
            return prediction.mean.detach().numpy(), prediction.variance.sqrt().detach().numpy()


def ip_likelihood(theta, full = False, plugin = False) :
   
    if full :
        
        pred = forward(theta)
        
        return (2*math.pi*eps_l**2)**(-0.5) * np.exp(- (pred-meas)**2/(2*eps_l**2) )
    
    else :
        
        # step needed due to surrogate model behaviour
        theta = theta.reshape( (-1, 1) )
    
        pred, std = GP.predict(theta)
        
        if plugin :
            cov = eps_l**2
        else :
            cov = std**2 + eps_l**2
        
        return (2*math.pi*cov)**(-0.5) * np.exp(- (pred-meas)**2/(2*cov) )
    


param_space = { 
    'min' : np.array([0.0]), 
    'max' : np.array([2.0])
}

solv_params = {
    'FE dim': 1,
    'FE order': 1,
    'solver order' : 1
}
def forward(theta, ) :
    
    return (1 - theta + theta**2 )* np.sin(4 * theta + 1)

np.random.seed(42)
eps_l = np.array([0.015])

theta_true = np.array([0.6])

meas = forward(theta_true )+ np.random.normal(0, eps_l)
meas = meas.reshape((-1,))

x = np.array([0.15,0.67,1.34, 1.87, 0.39, 1.53])
train_x = torch.tensor( x, dtype = torch.float32)
y = np.random.normal(forward(x), 0.3)
train_y = torch.tensor( y, dtype = torch.float32)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
GP = ExactGPModel(train_x, train_y, likelihood)
GP.fit()

test = np.linspace(-0.3,2, 1000)
x_test = torch.tensor(test, dtype = torch.float32)
mean, std = GP.predict(x_test)
GP = ExactGPModel(train_x, train_y, likelihood)

plt.figure(figsize = (12,7))
plt.plot(test, -forward(test), 'black', label="True function", lw = 3)
# plt.axvline(theta_true)
# plt.axhline(meas)
# plt.axhline(meas + 2 * eps_l, linestyle="dashdot")
# plt.axhline(meas - 2 * eps_l, linestyle="dashdot")

plt.plot(test, -mean, 'navy', label="Predictive mean", lw = 3)
plt.fill_between(
    test.ravel(),
    - mean - 1.96 * std,
    - mean + 1.96 * std,
    alpha=0.3,
    label=r"95% confidence interval",
    facecolor= 'cyan'
)

plt.scatter(x, -y, color = 'blue', marker = 'o', label= 'Training points')
#plt.title('True model vs GP')
plt.legend(loc = 'lower center', prop={'size': 16})
plt.axis('off')

plt.savefig(path+"model_comparison", format = 'svg', transparent = True)
plt.close()

# true_post = ip_likelihood(test, full = True)
# true_post = true_post/true_post.mean()
# approx_post = ip_likelihood(x_test, full = False)
# approx_post = approx_post/approx_post.mean()
# plug_post = ip_likelihood(x_test, full = False, plugin = True)
# plug_post = plug_post/plug_post.mean()

# plt.figure(figsize = (9,3))
# plt.plot(test, true_post, label='True')
# plt.plot(test, approx_post, label='Marginal')
# plt.plot(test, plug_post, label='Plug-in')
# plt.legend()
# plt.title('Posterior comparison')
# plt.savefig(path+"posterior_comparison", format = 'svg')
# plt.close()