import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import argparse

from IP.likelihoods import base_likelihood, GP_likelihood, lipschitz_likelihood
from IP.posteriors import Posterior
from IP.priors import GaussianPrior

from models.lipschitz import lipschitz_regressor
from models.GP_models.simple1DGP import SimpleGPModel

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
path = args.path

path = path + "/outputs/pictures/d1"
if not os.path.exists(path):
    os.makedirs(path)

param_space = { 
    'min' : np.array([0.0]), 
    'max' : np.array([1.0])
}

solv_params = {
    'FE dim': 1,
    'FE order': 1,
    'solver order' : 1
}
class Toy_forward() :
    dim =1
    dout =1
    def predict(self, theta):
        
        return theta + np.sin(np.pi * theta)/4

np.random.seed(42)
eps_l = np.array([0.015])

theta_true = np.array([0.45])
forward = Toy_forward()

meas = forward.predict(theta_true )+ np.random.normal(0, eps_l)
meas = meas.reshape((-1,))

x = np.array([0.1,0.3,0.7,0.9])
train_x = torch.tensor( x, dtype = torch.float32)
y = np.random.normal(forward.predict(x), 0.05)
train_y = torch.tensor( y, dtype = torch.float32)
LR_y = np.random.uniform(forward.predict(x) - 0.05, forward.predict(x) + 0.05)

#### initialize GP model
GP = SimpleGPModel(train_x, train_y, noise = 0.05*torch.ones((len(x))))
GP.fit()

#### initialize LR model
LR = lipschitz_regressor(dim=1,dout=1)
LR.fit(np.reshape(x,(-1,1)),LR_y.reshape((-1,1)), noise=0.05*np.ones(len(x)))

#### set up IP

prior = GaussianPrior(mean = 1/2, std = 1/3, dom=param_space)

true_likelihood = base_likelihood(meas, eps_l, forward)
GPR_likelihood = GP_likelihood(meas, eps_l, GP)
LR_likelihood = lipschitz_likelihood(meas, eps_l, LR)

true_posterior = Posterior(true_likelihood, prior)
GPR_posterior = Posterior(GPR_likelihood, prior)
LR_posterior = Posterior(LR_likelihood, prior)

#### test points

test = np.linspace(0,1, 1000)
gt_test = forward.predict(test).reshape((-1,))
mean_GP, std = GP.predict(test, return_std= True)
mean_LR, LB, UB = LR.predict(test.reshape((-1,1)), return_bds= True)

mean_GP = mean_GP.reshape((-1,))
std = std.reshape((-1,))
mean_LR = mean_LR.reshape((-1,))
LB = LB.reshape((-1,))
UB = UB.reshape((-1,))

true_post = true_posterior.prob(test).reshape((-1,))
GP_marg_post = GPR_posterior.prob(test).reshape((-1,))
GP_plug_post = GPR_posterior.prob(test, plug_in=True).reshape((-1,))
LR_marg_post = LR_posterior.prob(test).reshape((-1,))
LR_plug_post = LR_posterior.prob(test, plug_in=True).reshape((-1,))

GP_marg_post = GP_marg_post / np.mean(GP_marg_post)
GP_plug_post = GP_plug_post / np.mean(GP_plug_post)
LR_marg_post = LR_marg_post / np.mean(LR_marg_post)
LR_plug_post = LR_plug_post / np.mean(LR_plug_post)
true_post = true_post / np.mean(true_post)


#### GPR model plot
plt.figure(figsize = (9,3))
plt.plot(test, gt_test, label="True model", linestyle="dotted")


plt.plot(test, mean_GP, label="GP mean")
plt.fill_between(
    test.ravel(),
    mean_GP - 1.96 * std,
    mean_GP + 1.96 * std,
    alpha=0.5,
    label=r"95% conf. int.",
)

plt.plot(x, y,"ok", label= 'tr. pts.')
plt.title('True model vs GPR surrogate')
line = [[(0, 0)]]
#plt.axvline(theta_true)
line = plt.axhline(meas, label = '${y^m \pm 2 \sigma}$' )
plt.axhline(meas + 2 * eps_l, linestyle="dashdot")
plt.axhline(meas - 2 * eps_l, linestyle="dashdot")

from matplotlib.legend_handler import HandlerBase


class CustomHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        
        l1 = plt.Line2D([x0,y0+width], [0.9*height,0.9*height], 
                                                linestyle='dashdot',)
        l2 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height],)
        l3 = plt.Line2D([x0,y0+width], [0.1*height,0.1*height], 
                                                linestyle='dashdot',)
        patch = [l3, l2, l1]
        return patch
    
plt.legend(handler_map={line: CustomHandler()})

plt.savefig(path+"/GP_model_comparison.png", format = 'png')
plt.close()

#### LR model plot
plt.figure(figsize = (9,3))
plt.plot(test, gt_test, label="True model", linestyle="dotted")


plt.plot(test, mean_LR, label="GP mean")
plt.fill_between(
    test.ravel(),
    LB,
    UB,
    alpha=0.5,
    label=r"Uniform conf. int.",
)

plt.plot(x, LR_y,"ok", label= 'tr. pts.')
plt.title('True model vs LR surrogate')
line = [[(0, 0)]]
#plt.axvline(theta_true)
line = plt.axhline(meas, label = '${y^m \pm 2 \sigma}$' )
plt.axhline(meas + 2 * eps_l, linestyle="dashdot")
plt.axhline(meas - 2 * eps_l, linestyle="dashdot")
    
plt.legend(handler_map={line: CustomHandler()})

plt.savefig(path+"/LR_model_comparison.png", format = 'png')
plt.close()


plt.figure(figsize = (9,3))
plt.plot(test, true_post, label='True')
plt.plot(test, GP_marg_post, label='Marginal')
plt.plot(test, GP_plug_post, label='Plug-in')
plt.legend()
plt.title('Posterior comparison for GPR')
plt.savefig(path+"/GP_posterior_comparison.png", format = 'png')
plt.close()

plt.figure(figsize = (9,3))
plt.plot(test, true_post, label='True')
plt.plot(test, LR_marg_post, label='Marginal')
plt.plot(test, LR_plug_post, label='Plug-in')
plt.legend()
plt.title('Posterior comparison for LR')
plt.savefig(path+"/LR_posterior_comparison.png", format = 'png')
plt.close()