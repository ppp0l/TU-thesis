#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:13:57 2024

@author: pvillani
"""
import gpytorch
import torch

class SimpleGPModel(gpytorch.models.ExactGP):
    dim = 1
    dout = 1
    
    def __init__(self, train_x, train_y, noise):
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise**2)
        super(SimpleGPModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.noise = noise**2
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
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f ' % (
                i + 1, 200, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
            ), 
                  end="\r",
                  flush = True)
            
            optimizer.step()

    def predict(self, theta, return_std = False):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32)
        self.eval()
        self.likelihood.eval()
        # Make prediction
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            prediction = self(theta)
            mean =prediction.mean.detach().numpy()
            std = prediction.variance.sqrt().detach().numpy()

        mean = mean.reshape( (-1,1))
        std = std.reshape( (-1,1) )
        if return_std :
            return mean, std
        return mean