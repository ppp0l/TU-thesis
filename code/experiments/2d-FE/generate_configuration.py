import numpy as np

import json
import argparse
import os

dim = 2

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
args = parser.parse_args()
path = args.path

if not os.path.exists(path + f"/data/d{dim}"):
    os.makedirs(path + f"/data/d{dim}/")


from models.adaBeam import Adaptive_beam 
from utils.utils import reproducibility_seed

reproducibility_seed(seed=7856)

n_meas = 4

fm = Adaptive_beam(path)

# IP parameters
meas_std = 0.03

domain_upper_bound = np.ones(dim)
domain_lower_bound = np.zeros(dim)

domain_shift = 1/2

prior_mean = np.zeros(dim) + domain_shift
prior_std = 1/4

gt = np.random.normal(prior_mean - domain_shift, prior_std, (n_meas,dim))
while not np.all(np.abs(gt) <= domain_shift):
    out = gt[np.abs(gt) <= domain_shift]
    gt[np.abs(gt) <= domain_shift] = np.random.normal(prior_mean, prior_std, out.shape)

gt += domain_shift
    
pred = fm.predict(gt)

# adaptive training parameters
sample_every = 5
n_it = sample_every * 5
points_per_it = 1
n_init = 10
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
            "n_walkers": 64,
            "sample_every": sample_every,
            "init_samples": 200,
            "final_samples": 700,
        },
        "forward_model_config": {
            "FE_cost": FE_cost,
        },
    }
    configurations.append(config_dict)

json.dump(configurations, open(path + f"/data/d{dim}/config.json", "w"), indent=4)