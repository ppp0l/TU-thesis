import numpy as np

import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
path = args.path

if not os.path.exists(path + "/data/d3"):
    os.makedirs(path + "/data/d3/")


from models.forward import forward_model
from utils.utils import reproducibility_seed

reproducibility_seed(seed=7856)

dim = 3
n_meas = 5

fm = forward_model(dim = dim)

# IP parameters
meas_std = 0.01

domain_bound = 1/2

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
sample_every = 3
n_it = sample_every * 5
points_per_it = 1
n_init = 5
default_tol_fixed = 0.001
default_tol_ada = 0.005
FE_cost = 1
budget = (n_init + points_per_it * n_it) * (default_tol_ada)**(-FE_cost)

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
            "n_it": n_it,
            "points_per_it": points_per_it,
            "default_tol_fixed": default_tol_fixed,
            "default_tol_ada": default_tol_ada,
            "budget": budget,
        },
        "sampling_config": {
            "n_walkers": 32,
            "sample_every": sample_every,
            "init_samples": 200,
            "final_samples": 700,
        },
        "forward_model_config": {
            "FE_cost": FE_cost,
        },
    }
    configurations.append(config_dict)

json.dump(configurations, open(path + "/data/d3/config.json", "w"), indent=4)