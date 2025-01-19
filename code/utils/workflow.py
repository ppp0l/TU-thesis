#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:32:49 2024

@author: pvillani
"""
import csv
import os
import numpy as np
import ast
import torch

class Manager() :
    def __init__(self, path : str, dimension : int, try_per_param : int = 5):
        
        self.path = path
        
        self.try_per_param = try_per_param

        self.dimension = dimension
        
    def read_parameters(self, type_res = "full"):
        
        config_path = self.path+f'/data/d{self.dimension}'
        
        with open(config_path+'/configurations.csv', 'r', newline='') as file:
            parameter_reader = csv.DictReader(file, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
            
            
            for i,parameters in enumerate(parameter_reader):
                
                # if i == 0 : continue
                # if i>1 and parameters["target"] != "L2" :
                #     continue
                
                res_path= self.path+f'/outputs/d{self.dimension}{parameters["target"]}/FE{parameters["FE cost"]}'
                # parametri sono numerati
                parameters['number'] = i
                parameters['results path'] = res_path
                parameters['value'] = np.array(ast.literal_eval(parameters['value']))
                parameters["n_init"] = int(parameters["n_init"])
                if "sensors" in parameters :
                    parameters['sensors'] = np.array(ast.literal_eval(parameters['sensors']))

                if "meas_time" in parameters :
                    parameters['meas_time'] = np.array(ast.literal_eval(parameters['meas_time']))


                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                
                if not os.path.exists(res_path + os.sep + type_res + f'_res_{i}.csv' ):
                    # se non abbiamo mai fatto andare questo parametro, andata
                    parameters['seed'] = 0
                    self.configuration = parameters
                    return parameters
                
                with open(res_path + os.sep + type_res + f'_res_{i}.csv' , 'r', newline='') as res_file :
                    num_lines = sum(1 for line in res_file) -1
                    if num_lines < self.try_per_param :
                        # se questo parametro e' andato meno di tot volte, andata
                        parameters['seed'] = num_lines
                        self.configuration = parameters
                        return parameters
        
        raise ValueError('All parameters in configurations.csv were tested enough times. Add more!')
    
    def save_results(self, results, type_res):
        parameters = self.configuration
        
        num = parameters['number']
        seed = parameters['seed']

        res_file_path = parameters['results path'] + '/' + type_res + f'_res_{num}.csv'

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

    def state_saver(self, type_res, nit, W, model, samples = None) :
        parameters = self.configuration

        state_path = self.path+f'/data/d{self.dimension}{parameters["target"]}/states/FE{parameters["FE cost"]}'
        num = parameters['number']
        seed = parameters['seed']

        state_path += f'/num{num}_{seed}'

        if not os.path.exists(state_path):
            os.makedirs(state_path)

        # save checkpoint parameters
        with open(state_path + '/' + type_res + '_checkpoint_data.csv', 'a', newline='') as file :
            writer = csv.writer(file, delimiter = ',', quoting =csv.QUOTE_NONNUMERIC)
            writer.writerow([nit,W])

        # if we have some samples, save them
        if nit > 0 :
            if type_res == 'lhs' :
                pass
            else:   
                with open(state_path + f'/it{nit}_{type_res}_samples.npy', 'wb') as file:
                    np.save(file, samples)

        # save current state of GP
        GP_file = state_path + f'/it{nit}_{type_res}_model.pth'
        torch.save(model.state_dict(), GP_file)

        # save training set
        tr_dict = {
            'tr_p' : model.training_p,
            'tr_y' : model.training_y,
            'errs' : model.errors
        }
        tr_file = state_path + f'/it{nit}_{type_res}_training_set.pth'
        torch.save(tr_dict, tr_file)
        
        
    def save_other_metric(self, metric, type_res) :

        print("Saving other metric...")
        
        parameters = self.configuration
        
        num = parameters['number'] % 5
        res_path = parameters['results path'] + os.sep + ".other_metric"
        
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        res_file_path =  res_path + os.sep + type_res + '.csv'
        
        value, _ = metric(other = True)
        
        if not os.path.exists(res_file_path ):
            data = [[value]]
        else :
            with open(res_file_path, 'r', newline='') as res_file :
                reader = csv.reader(res_file, delimiter=',', quoting = csv.QUOTE_NONNUMERIC )
                data = list(reader)
                
            if len(data) > num:
                data[num].append(value)
            else : 
                len_d = len(data)
                for i in range(len_d, num) :
                    data.append(dict())
                data.append([value])
        
        with open(res_file_path, 'w', newline='') as res_file :
            writer = csv.writer(res_file, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)

            for line in data:
                writer.writerow(line)

        print("Done")
        

    
    def state_loader(self, model, type_res, initial = False) :
        parameters = self.configuration

        state_path = self.path+f'/data/d{self.dimension}{parameters["target"]}/states/FE{parameters["FE cost"]}'
        num = parameters['number']
        seed = parameters['seed']

        state_path += f'/num{num}_{seed}'

        with open(state_path + '/' +  type_res + '_checkpoint_data.csv', 'r', newline='') as file :
            reader = csv.reader(file, delimiter = ',', quoting =csv.QUOTE_NONNUMERIC)
            
            for row in reader :
                pars = row
                if initial :
                    break
            nit = int(pars[0])
            W = pars[1]
    
            
        
        GP_file = state_path + f'/it{nit}_{type_res}_model.pth'
        GP_state_dict = torch.load(GP_file)

        tr_file = state_path + f'/it{nit}_{type_res}_training_set.pth'
        tr_set = torch.load(tr_file)
        train_p = tr_set['tr_p']
        train_y = tr_set['tr_y']
        errors = tr_set['errs']

        model.fit( train_p, train_y, noise = errors**2)
        model.load_state_dict(GP_state_dict)
        
        if nit > 0 :
            if type_res == 'lhs' :
                samples = None
            else:   
                with open(state_path + f'/it{nit}_{type_res}_samples.npy', 'rb') as file:
                    samples = np.load(file)
        else :
            samples = None

        return nit, W, samples