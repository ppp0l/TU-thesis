import numpy as np

import json
import argparse
import os

dim = 2

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
parser.add_argument("--eval_model", type=bool, default=False, help="Evaluate the model to generate the data or not")
args = parser.parse_args()
path = args.path

kaskade_path = path + f"/data/d{dim}/kaskade"

if not os.path.exists(kaskade_path):
    os.makedirs(kaskade_path+"/")
    args.eval_model = True


from models.adaBeam import Adaptive_beam 
from utils.utils import reproducibility_seed

reproducibility_seed(seed=7856)

fm = Adaptive_beam(path, mesh = kaskade_path + "gt_mesh.vtu", adaptive = False)

# IP parameters
meas_std = 0.03

domain_upper_bound = np.ones(dim)
domain_lower_bound = np.zeros(dim)

prior_mean = np.ones(dim)/2
prior_std = 1/6

n_meas = 1
gt = np.array([ [2.7e11, 0.3]])

    
if args.eval_model:
    gt = fm.scale_parameters(gt)
    pred = fm.predict(gt, tols= [0])
    np.save(path + f"/data/d{dim}/gt_measurements.npy", pred)
else :
    pred = np.load(path + f"/data/d{dim}/gt_measurements.npy")

# adaptive training parameters
sample_every = 2
points_per_it = 1
n_init = 3
default_tol = 0.03
threshold = meas_std**2 * fm.dout / 20
conv_ratio = 1/4
max_iter = 6
FE_cost = 1.5

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
            "n_walkers": 16,
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