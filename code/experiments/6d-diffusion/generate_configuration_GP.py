import numpy as np

import json
import argparse
import os

dim = 6

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
path = args.path

if not os.path.exists(path + f"/data/d{dim}"):
    os.makedirs(path + f"/data/d{dim}/")


from models.forward import forward_model
from utils.utils import reproducibility_seed

reproducibility_seed(seed=7856)

n_meas = 4

fm = forward_model(dim = dim)

# IP parameters
meas_std = 0.02

domain_bound = 1

domain_upper_bound = np.ones(dim) * domain_bound
domain_lower_bound = -np.ones(dim) * domain_bound

prior_mean = np.zeros(dim)
prior_std = 1/4

gt = np.random.normal(prior_mean, prior_std, (n_meas,dim))
while not np.all(np.abs(gt) <= domain_bound):
    out = gt[np.abs(gt) <= domain_bound]
    gt[np.abs(gt) <= domain_bound] = np.random.normal(prior_mean, prior_std, out.shape)
    
pred = fm.predict(gt)

# adaptive training parameters
sample_every = 4
points_per_it = 1
n_init = 13
default_tol = 0.03
threshold = meas_std**2 * fm.dout / 40
conv_ratio = 1/8
max_iter = 6
FE_cost = 1

configurations = []
for i in range(n_meas):
    ground_truth = gt[i].tolist()
    measurement = pred[i]+ np.random.normal(0, meas_std, pred[i].shape)
    config_dict = {
        "IP_config" : {
            "ground_truth": ground_truth,
            "measurement": measurement.tolist(),
            "measurement_std": meas_std,
            "domain_upper_bound": domain_upper_bound.tolist(),
            "domain_lower_bound": domain_lower_bound.tolist(),
            "prior_mean": prior_mean.tolist(),
            "prior_std": prior_std,
        },
        "training_config": {
            "n_init": n_init,
            "points_per_it": points_per_it,
            "default_tol": default_tol,
            "max_iter": max_iter * sample_every,
            "threshold": threshold,
            "conv_ratio": conv_ratio,
        },
        "sampling_config": {
            "n_walkers": 64,
            "sample_every": sample_every,
            "n_sample": 250,
            "n_burn": 250,
        },
        "forward_model_config": {
            "FE_cost": FE_cost,
        },
    }
    configurations.append(config_dict)

json.dump(configurations, open(path + f"/data/d{dim}/GP_config.json", "w"), indent=4)