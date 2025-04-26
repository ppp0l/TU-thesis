#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:32:49 2024

@author: pvillani
"""
import csv
import json
import os
import numpy as np
import ast
import torch

def process_configuration(configuration):
    """
    Helper function processing the configuration dictionary to adjust types of certain entries.
    """

    for key, value in configuration["IP_config"].items():
        if isinstance(value, list):
            # Convert list to numpy array
            configuration["IP_config"][key] = np.array(value)

    configuration["training_config"]["n_init"] = int(configuration["training_config"]["n_init"])
    #configuration["training_config"]["n_it"] = int(configuration["training_config"]["n_it"])
    configuration["training_config"]["points_per_it"] = int(configuration["training_config"]["points_per_it"])

    configuration["sampling_config"]["n_walkers"] = int(configuration["sampling_config"]["n_walkers"])
    configuration["sampling_config"]["sample_every"] = int(configuration["sampling_config"]["sample_every"])
    configuration["sampling_config"]["n_sample"] = int(configuration["sampling_config"]["n_sample"])
    configuration["sampling_config"]["n_burn"] = int(configuration["sampling_config"]["n_burn"])

class Manager() :
    def __init__(self, path : str, dimension : int, try_per_param : int = 5):
        
        self.path = path
        
        self.try_per_param = try_per_param

        self.dimension = dimension
        
    def read_configuration(self, type_res = "AGP"):
        
        config_file = self.path+f'/data/d{self.dimension}'
        if "GP" in type_res :
            config_file += '/GP_config.json'
        else :
            config_file += '/LR_config.json'
        
        with open(config_file, 'r', newline='') as file:
            configurations = json.load(file)
            
            
            for i,configuration in enumerate(configurations):
                
                res_path= self.path+f'/outputs/d{self.dimension}'
                # parametri sono numerati
                configuration['number'] = i
                configuration['res_path'] = res_path

                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                
                if not os.path.exists(res_path + os.sep + type_res + f'_res_{i}.csv' ):
                    # se non abbiamo mai fatto andare questo parametro, andata
                    configuration['seed'] = 0
                    process_configuration(configuration)
                    self.configuration = configuration
                    return configuration
                
                with open(res_path + os.sep + type_res + f'_res_{i}.csv' , 'r', newline='') as res_file :
                    num_lines = sum(1 for line in res_file) -1
                    if num_lines < self.try_per_param :
                        # se questo parametro e' andato meno di tot volte, andata
                        configuration['seed'] = num_lines
                        process_configuration(configuration)
                        self.configuration = configuration
                        return configuration
        
        raise ValueError('All configurations in config.json were tested enough times. Add more!')
    
    def save_results(self, results, type_res):
        configuration = self.configuration
        
        num = configuration['number']
        seed = configuration['seed']

        res_file_path = configuration['res_path'] + '/' + type_res + f'_res_{num}.csv'

        res_dict = {round(results['W'][i]) : results['target'][i] for i in range(len(results['W']))}
        if not os.path.exists(res_file_path ):

            fieldnames = np.round(results['W'])
            data = [dict() for _ in range(seed)]
            data.append(res_dict)
        else :
            with open(res_file_path, 'r', newline='') as res_file :
                reader = csv.DictReader(res_file, delimiter=',', quoting = csv.QUOTE_NONNUMERIC )
                fieldnames = reader.fieldnames
                data = list(reader)
            name_set = set(fieldnames)
            name_set = name_set.union(np.round(results['W']) )
            fieldnames = np.sort(np.array(list(name_set), dtype=int))


            if len(data) < seed + 1: 
                len_d = len(data)
                for i in range(len_d, seed) :
                    data.append(dict())
                data.append(res_dict)
            else : 
                data[seed] = { **data[seed] , **res_dict}
        
        with open(res_file_path, 'w', newline='') as res_file :
            writer = csv.DictWriter(res_file, fieldnames, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)

            writer.writeheader()
            for line in data:
                writer.writerow(line)


    def state_saver(self, type_res, nit, W, training_set, model, samples) :
        configuration = self.configuration

        state_path = self.path+f'/data/d{self.dimension}/'+type_res
        num = configuration['number']
        seed = configuration['seed']

        state_path += f'/num{num}_{seed}'

        if not os.path.exists(state_path):
            os.makedirs(state_path)

        # save checkpoint parameters
        with open(state_path + '/checkpoint_data.csv', 'a', newline='') as file :
            writer = csv.writer(file, delimiter = ',', quoting =csv.QUOTE_NONNUMERIC)
            writer.writerow([nit,W])

        # save samples
        with open(state_path + f'/it{nit}_samples.npy', 'wb') as file:
            np.save(file, samples)

        # save current state of GP
        model_file = state_path + f'/it{nit}_model.pth'
        torch.save(model.state_dict(), model_file)

        try: 
            if model.model.likelihood.has_task_noise :
                task_cov = model.model.likelihood.task_noise.detach().numpy()
                task_cov_file = state_path + f'/it{nit}_task_cov.npy'
                np.save(task_cov_file, task_cov)
        except AttributeError:
            pass

        # save training set
        tr_file = state_path + f'/it{nit}_training_set.pth'
        torch.save(training_set, tr_file)
        

    def state_loader(self, model, type_res, initial = False) :
        configuration = self.configuration

        state_path = self.path+f'/data/d{self.dimension}/'+type_res
        num = configuration['number']
        seed = configuration['seed']

        state_path += f'/num{num}_{seed}'

        with open(state_path + '/checkpoint_data.csv', 'r', newline='') as file :
            reader = csv.reader(file, delimiter = ',', quoting =csv.QUOTE_NONNUMERIC)
            
            for row in reader :
                pars = row
                if initial :
                    break
            nit = int(pars[0])
            W = pars[1]            
        
        model_file = state_path + f'/it{nit}_model.pth'
        state_dict = torch.load(model_file)

        tr_file = state_path + f'/it{nit}_training_set.pth'
        tr_set = torch.load(tr_file)
        train_p = tr_set['train_p']
        train_y = tr_set['train_y']
        errors = tr_set['errors']

        if os.path.exists(state_path + f'/it{nit}_task_cov.npy'):
            task_cov = np.load(state_path + f'/it{nit}_task_cov.npy')
            model.fit(train_p, train_y, errors, likelihood_has_task_noise=True, likelihood_task_noise=task_cov)
        else :
            model.fit( train_p, train_y, noise = errors)
        model.load_state_dict(state_dict)
        
        with open(state_path + f'/it{nit}_samples.npy', 'rb') as file:
            samples = np.load(file)
        return nit, W, samples