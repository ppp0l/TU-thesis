import os
import argparse

import numpy as np
from copy import deepcopy

from utils.workflow import process_configuration
from utils.postprocessing import gt_samples, gt_MAP, compute_surrogate_MAP
from utils.utils import latin_hypercube_sampling as lhs
from utils.plots import corner_plot

import json
import csv
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path for data")
parser.add_argument("--dim", type=int, default=3, help="Dimension of the problem")
parser.add_argument("--type_surr", type=str, default="GP", help="Type of surrogate model")
parser.add_argument("--include_gt", type=bool, default=False, help="Include ground truth for comparison")
parser.add_argument("--proc_samples", type=bool, default=False, help="Process samples")
parser.add_argument("--proc_MAP", type=bool, default=False, help="Process MAP")
parser.add_argument("--proc_n_pts", type=bool, default=False, help="Process number of training points")
args = parser.parse_args()
path = args.path
dim = args.dim
type_surr = args.type_surr

data_path = path + f"/data/d{dim}"

config_file = data_path + "/GP_config.json"

configs = json.load(open(config_file, 'r'))

gt_path = data_path + "/postprocessing/gt"

pic_path = path + f"/outputs/pictures/d{dim}"

if not os.path.exists(gt_path):
    os.makedirs(gt_path+"/")
    
for i, config in enumerate(configs):
    process_configuration(config)
    config["number"] = i

    domain = {
        "min": config["IP_config"]["domain_lower_bound"],
        "max": config["IP_config"]["domain_upper_bound"]
        }
    
    means = {}
    stds = {}
    MAPs = {}
    n_train_pts= {}
    samples_dict = {}

    if args.include_gt :
        if args.proc_samples :
            try :
                true_samples = np.load(gt_path + f"/meas{i}_samples.npy")
            except :
                true_samples = gt_samples(dim, config)
                np.save(gt_path + f"/meas{i}_samples.npy", true_samples)
            starts = true_samples[ np.random.randint(0, len(true_samples), 3*dim)]

            # remove outliers
            tr_mean = np.mean(true_samples, axis = 0)
            tr_std  = np.sqrt( np.mean(true_samples**2, axis = 0 )- tr_mean**2)
            cleaned_true = true_samples[np.all( true_samples - tr_mean < 4*tr_std, axis = 1)]

            means = {"Ground truth": tr_mean}
            stds = {"Ground truth": tr_std}
            samples_dict["Ground truth"] = cleaned_true
        else :
            starts = lhs(domain["min"], domain["max"], 3*dim)
        
        if args.proc_MAP :
            try :
                true_map = np.load(gt_path + f"/meas{i}_MAP.npy")
            except :
                
                true_map = gt_MAP(dim, config, starts)
                np.save(gt_path + f"/meas{i}_MAP.npy", true_map)

            MAPs = {"Ground truth": true_map}

    for seed in range(5) :
        samples = deepcopy(samples_dict)
        
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
            
            if args.proc_samples :
                with open(curr_path + f'/it{nit}_samples.npy', 'rb') as file:
                    curr_samples = np.load(file)

                samp_mean = np.mean(curr_samples, axis = 0)
                samp_std  = np.sqrt( np.mean(curr_samples**2, axis = 0 )- samp_mean**2)
                cleaned_samples = curr_samples[np.all( curr_samples - samp_mean < 4*samp_std, axis = 1)]

                samples[type_run+type_surr] = cleaned_samples
                
                min_len = min( [len(cleaned_samples), min_len])
                starts = curr_samples[ np.random.randint(0, len(curr_samples), 3*dim)]

                try :
                    means[type_run+type_surr].append(samp_mean)
                    stds[type_run+type_surr].append(samp_std)
                except :
                    means[type_run+type_surr] = [samp_mean]
                    stds[type_run+type_surr] = [samp_std]

            if args.proc_MAP :

                model_file = curr_path + f'/it{nit}_model.pth'
                state_dict = torch.load(model_file)

                tr_file = curr_path + f'/it{nit}_training_set.pth'
                tr_set = torch.load(tr_file)

                try :
                    cov_est = np.load(curr_path + f'/it{nit}_task_cov.npy')
                except :
                    cov_est = None

                MAP = compute_surrogate_MAP(config, starts, type_surr, tr_set, state_dict, cov_est)

                try :
                    MAPs[type_run+type_surr].append(MAP)
                except :
                    MAPs[type_run+type_surr] = [MAP]

            if args.proc_n_pts :
                n_train = len(tr_set['train_p'])
                try :
                    n_train_pts[type_run+type_surr].append(n_train)
                except :
                    n_train_pts[type_run+type_surr] = [n_train]
                    
            
            
        if args.proc_samples :
            for key in samples.keys() :
                samples[key] = samples[key][:min_len]

            
            corner_plot( list(samples.values()), 
                        labels = list(samples.keys()), 
                        colors = ["blue", "orange", "green", "red", "purple"],
                        savepath = pic_path + f"/corner_plots/{type_surr}/run{i}_{seed}.png",
                        saveformat = "png",
                        title = f"Corner plot for {type_surr} surrogate, measurement {i} and seed {seed}",
                        domain = domain,
                        dim = dim,
                        )
    if args.proc_samples :
        torch.save(means, data_path +f"/postprocessing/meas{i}_{type_surr}_means.pth")
        torch.save(stds, data_path +f"/postprocessing/meas{i}_{type_surr}_stds.pth")
    if args.proc_MAP :
        torch.save(MAPs, data_path +f"/postprocessing/meas{i}_{type_surr}_MAPs.pth")
    if args.proc_n_pts :
        torch.save(n_train_pts, data_path +f"/postprocessing/meas{i}_{type_surr}_n_train.pth")

        

            
            
            




    
    

