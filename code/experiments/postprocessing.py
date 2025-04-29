import os
import argparse

import numpy as np

from models.forward import forward_model

from IP.priors import GaussianPrior
from IP.likelihoods import base_likelihood
from IP.posteriors import Posterior

from utils.workflow import process_configuration
from utils.postprocessing import gt_samples, gt_MAP
from utils.utils import latin_hypercube_sampling as lhs
from utils.plots import corner_plot

import json
import csv
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
parser.add_argument("--dim", type=int, default=3, help="Dimension of the problem")
parser.add_argument("--type_surr", type=str, default="GP", help="Type of surrogate model")
args = parser.parse_args()
path = args.path
dim = args.dim
type_surr = args.type_surr

data_path = path + f"/data/d{dim}"

config_file = data_path + "/GP_config.json"

configs = json.load(open(config_file, 'r'))

gt_path = data_path + "/gt"

pic_path = path + f"/outputs/pictures/d{dim}"

if not os.path.exists(gt_path):
    os.makedirs(gt_path+"/")
    
for i, config in enumerate(configs):
    process_configuration(config)
    config["number"] = i

    try :
        true_samples = np.load(gt_path + f"/meas{i}_samples.npy")
    except :
        true_samples = gt_samples(dim, config)
        np.save(gt_path + f"/meas{i}_samples.npy", true_samples)
    
    try :
        true_map = np.load(gt_path + f"/meas{i}_MAP.npy")
    except :
        starts = true_samples[ np.random.randint(0, len(true_samples), 3*dim)]
        true_map = gt_MAP(dim, config, starts)
        np.save(gt_path + f"/meas{i}_MAP.npy", true_map)
    
    # remove outliers
    tr_mean = np.mean(true_samples, axis = 0)
    tr_std  = np.sqrt( np.mean(true_samples**2, axis = 0 )- tr_mean**2)
    cleaned_true = true_samples[np.all( true_samples - tr_mean < 4*tr_std, axis = 1)]

    domain = {
        "min": config["IP_config"]["domain_lower_bound"],
        "max": config["IP_config"]["domain_upper_bound"]
        }

    for seed in range(5) :
        samples = {"Ground truth": cleaned_true}
        min_len = len(cleaned_true)
        for type_run in ["A", "LHS", "posAd", "rand"] :
            curr_path = data_path + f"/{type_run}{type_surr}/num{i}_{seed}"
            if not os.path.exists(curr_path):
                continue
            with open(curr_path + '/checkpoint_data.csv', 'r', newline='') as file :
                reader = csv.reader(file, delimiter = ',', quoting =csv.QUOTE_NONNUMERIC)
                pars =[0,0]
                for row in reader :
                    pars = row
                nit = int(pars[0])
            
            model_file = curr_path + f'/it{nit}_model.pth'
            state_dict = torch.load(model_file)

            tr_file = curr_path + f'/it{nit}_training_set.pth'
            tr_set = torch.load(tr_file)
            train_p = tr_set['train_p']
            train_y = tr_set['train_y']
            errors = tr_set['errors']

            # if os.path.exists(curr_path + f'/it{nit}_task_cov.npy'):
            #     task_cov = np.load(curr_path + f'/it{nit}_task_cov.npy')
            #     model.fit(train_p, train_y, errors, likelihood_has_task_noise=True, likelihood_task_noise=task_cov)
            # else :
            #     model.fit( train_p, train_y, noise = errors)
            # model.load_state_dict(state_dict)
            
            with open(curr_path + f'/it{nit}_samples.npy', 'rb') as file:
                curr_samples = np.load(file)
            
            samp_mean = np.mean(curr_samples, axis = 0)
            samp_std  = np.sqrt( np.mean(curr_samples**2, axis = 0 )- samp_mean**2)
            cleaned_samples = curr_samples[np.all( curr_samples - samp_mean < 4*samp_std, axis = 1)]

            samples[type_run+type_surr] = cleaned_samples
            min_len = min( [len(cleaned_samples), min_len])

        for key in samples.keys() :
            samples[key] = samples[key][:min_len]
            
        corner_plot( list(samples.values()), 
                     labels = list(samples.keys()), 
                     colors = ["blue", "orange", "green", "red", "purple"],
                     savepath = pic_path + f"/corner_plots/{type_surr}/run{i}_{seed}.png",
                     saveformat = "png",
                     title = f"Corner plot for {i} and {seed}",
                     domain = domain,
                     dim = dim,
                    )
        

        

            
            
            




    
    

