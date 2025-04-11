import gc 

import numpy as np


import AL.L2_GP
import AL.exp_err_red
import AL.position.GP
import AL.position.lips
import AL.tolerance
import AL.tolerance.GP
import AL.tolerance.lips
from utils.workflow import Manager

from models.surrogate import Surrogate
from models.forward import forward_model

from IP.posteriors import Posterior

import AL
from utils.plots import corner_plot

def L1_target(surrogate, samples):

    _, LB, UB = surrogate.predict(samples, return_bds=True)

    return AL.exp_err_red.L1_err(LB, UB).mean()

def L2_target(surrogate, samples):

    _, std = surrogate.predict(samples, return_std=True)

    return AL.L2_GP.L2_approx(std**2).mean()

def solve_pos_prob( points_per_it, param_space, default_tol, surrogate, samples, FE_cost):

    if "GP" in str(type(surrogate)) :
        _, std_samples = surrogate.predict(samples, return_std=True)
        std_samples = std_samples.mean(axis = 0)
        return AL.position.GP.solve_pos_prob(points_per_it, param_space, default_tol, surrogate, samples, std_samples, FE_cost,)
    
    else :

        return AL.position.lips.solve_pos_prob(points_per_it, param_space, default_tol, surrogate, samples)
    
def solve_acc_prob(candidates, W, surrogate, samples, FE_cost):

    if "GP" in str(type(surrogate)) :
        _, std_samples = surrogate.predict(samples, return_std=True)
        std_samples = std_samples.mean(axis = 0)
        return AL.tolerance.GP.solve_acc_prob(candidates, W, surrogate, samples, std_samples, FE_cost)
    else :

        return AL.tolerance.lips.solve_acc_prob(candidates, W, surrogate, samples, FE_cost)

def run(run_type : str, 
        training_set : dict,
        surrogate : Surrogate,
        fm : forward_model,
        posterior : Posterior,
        workflow_manager : Manager,
        ) :
    
    match run_type :
        case "AGP" :
            run_adaptive(training_set, surrogate, fm, posterior, workflow_manager, run_type, L2_target)
        case "ALR" :
            run_adaptive(training_set, surrogate, fm, posterior, workflow_manager, run_type, L1_target)
        case "posAdGP" :
            run_fixed_tolerance(training_set, surrogate, fm, posterior, workflow_manager, run_type, L2_target)
        case "posAdLR" :
            run_fixed_tolerance(training_set, surrogate, fm, posterior, workflow_manager, run_type, L1_target)
        case "randGP" :
            run_random(training_set, surrogate, fm, posterior, workflow_manager, run_type, L2_target)
        case "randLR" :
            run_random(training_set, surrogate, fm, posterior, workflow_manager, run_type, L1_target)
        case _ :
            raise ValueError(f"Unknown run type: {run_type}")

    

def run_adaptive(training_set : dict, surrogate : Surrogate, fm : forward_model, posterior : Posterior, workflow_manager: Manager, run_type : str, target):
    configuration = workflow_manager.configuration

    param_space = fm.dom
    dim = fm.dim

    # initial training set
    train_p = training_set["train_p"]
    train_y = training_set["train_y"]
    errors = training_set["errors"]

    # active learning parameters
    training_config = configuration["training_config"]
    sampling_config = configuration["sampling_config"]

    points_per_it = training_config["points_per_it"]
    n_it = training_config["n_it"]
    sample_every = sampling_config["sample_every"]

    default_tol = training_config["default_tol_ada"]
    budget_ada = training_config["budget"]
    budget_per_it = budget_ada / n_it

    FE_cost = configuration["forward_model_config"]["FE_cost"]

    n_init_samples = sampling_config["init_samples"]
    n_final_samples = sampling_config["final_samples"]
    n_walkers = sampling_config["n_walkers"]
    samples = np.array([np.zeros(dim)])
    

    for i in range(n_it) :
        if i%sample_every == 0 :
            print("Sampling posterior...") 
            print()

            n_burn = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it)**2 )
            n_samples = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it))
            # sample posterior
            new_samples = posterior.sample_points(n_samples)

            # update sample chains
            if n_burn * n_walkers > len(samples) :
                samples = np.array([np.zeros(dim)])
            else :
                samples = samples[n_burn*n_walkers :]
            samples = np.concatenate( (samples, new_samples), axis = 0)
            print("Done.")
            print()
            shortened_samples = samples[::5]

        # monitor convergence
        W = np.sum(errors**(-FE_cost))

        curr_target = target(surrogate, shortened_samples)

        # save results
        if i%sample_every == 0 :
            workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
            training_set["train_p"] = train_p
            training_set["train_y"] = train_y
            training_set["errors"] = errors
            workflow_manager.state_saver(run_type, i, W, training_set, surrogate, samples)

        print("Rietriving candidates...")
        print()
        # position problem
        candidates = solve_pos_prob(points_per_it, param_space, default_tol, surrogate, shortened_samples, FE_cost )
        print("Done.")
        print()

        print("Optimizing tolerances...")
        print()
        # accuracy problem
        tolerances, new_pts, updated = solve_acc_prob(candidates, budget_per_it, surrogate, shortened_samples, FE_cost)
        print("Done.")
        print()

        new_tols = tolerances[len(train_p):]
        update_tols = tolerances[:len(train_p)]

        print("Evaluating model...")
        print()
        # evaluate model
        if np.any(updated) :
            train_y[updated], errors[updated] = fm.predict(train_p[updated], update_tols[updated])

        if len(new_pts) > 0:
            new_vals, new_errs = fm.predict(new_pts, new_tols)

            train_p =  np.concatenate((train_p, new_pts), axis = 0)
            train_y = np.concatenate((train_y, new_vals), axis = 0)
            errors = np.concatenate((errors, new_errs), axis = 0)

        print("Done.")
        print()

        print("Updating surrogate...")
        print()
        # update surrogate
        surrogate.fit(train_p, train_y, errors**2)
        print("Done.")
        print()

        new_target = target(surrogate, shortened_samples)


        print()
        print(f"Iteration {i}")
        print(f"Precedent target approx value: {curr_target}")
        print(f"Current target approx value: {new_target}")
        print(f"Points in the training set: {len(train_p)}")
        print()

        gc.collect()
    
    # export final round, save
    n_burn = n_walkers * n_samples
    n_samples = n_final_samples
    # sample posterior
    new_samples = posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

    shortened_samples = samples[::5]

    # monitor convergence
    W = np.sum(errors**(-FE_cost))

    curr_target = target(surrogate, shortened_samples)

    # save results
    workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
    workflow_manager.state_saver(run_type, n_it, W, training_set, surrogate, samples)

def run_fixed_tolerance(training_set : dict, surrogate : Surrogate, fm : forward_model, posterior : Posterior, workflow_manager: Manager, run_type : str , target):
    configuration = workflow_manager.configuration

    param_space = fm.dom
    dim = fm.dim

    # initial training set
    train_p = training_set["train_p"]
    train_y = training_set["train_y"]
    errors = training_set["errors"]

    # active learning parameters
    training_config = configuration["training_config"]
    sampling_config = configuration["sampling_config"]

    points_per_it = training_config["points_per_it"]
    n_it = training_config["n_it"]
    sample_every = sampling_config["sample_every"]

    default_tol = training_config["default_tol_fixed"]

    FE_cost = configuration["forward_model_config"]["FE_cost"]

    n_init_samples = sampling_config["init_samples"]
    n_final_samples = sampling_config["final_samples"]
    n_walkers = sampling_config["n_walkers"]
    samples = np.array([np.zeros(dim)])
    

    for i in range(n_it) :
        if i%sample_every == 0 :
            print("Sampling posterior...") 
            print()

            n_burn = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it)**2 )
            n_samples = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it))
            # sample posterior
            new_samples = posterior.sample_points(n_samples)

            # update sample chains
            if n_burn * n_walkers > len(samples) :
                samples = np.array([np.zeros(dim)])
            else :
                samples = samples[n_burn*n_walkers :]
            samples = np.concatenate( (samples, new_samples), axis = 0)
            print("Done.")
            print()
            shortened_samples = samples[::5]

        # monitor convergence
        W = np.sum(errors**(-FE_cost))

        _, std = surrogate.predict(shortened_samples, return_std=True)

        curr_target = target(surrogate, shortened_samples)

        # save results
        if i%sample_every == 0 :
            workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
            training_set["train_p"] = train_p
            training_set["train_y"] = train_y
            training_set["errors"] = errors
            workflow_manager.state_saver(run_type, i, W, training_set, surrogate, samples)

        print("Rietriving candidates...")
        print()
        # position problem
        new_pts = solve_pos_prob( points_per_it, param_space, default_tol, surrogate, shortened_samples, FE_cost )
        print("Done.")
        print()

        new_tols = default_tol * np.ones(len(new_pts))

        print("Evaluating model...")
        print()
        # evaluate model
        new_vals, new_errs = fm.predict(new_pts, new_tols)
        
        print("Done.")
        print()

        print("Updating surrogate...")
        print()
        # update surrogate
        train_p =  np.concatenate((train_p, new_pts), axis = 0)
        train_y = np.concatenate((train_y, new_vals), axis = 0)
        errors = np.concatenate((errors, new_errs), axis = 0)

        surrogate.fit(train_p, train_y, errors**2)
        print("Done.")
        print()

        new_target = target(surrogate, shortened_samples)

        print()
        print(f"Iteration {i}")
        print(f"Precedent target approx value: {curr_target}")
        print(f"Current target approx value: {new_target}")
        print(f"Points in the training set: {len(train_p)}")
        print()

        gc.collect()
    
    # export final round, save
    n_burn = n_walkers * n_samples
    n_samples = n_final_samples
    # sample posterior
    new_samples = posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

    shortened_samples = samples[::5]

    # monitor convergence
    W = np.sum(errors**(-FE_cost))
    
    curr_target = target(surrogate, shortened_samples)

    # save results
    workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
    workflow_manager.state_saver(run_type, n_it, W, training_set, surrogate, samples)

def run_random(training_set : dict, surrogate : Surrogate, fm : forward_model, posterior : Posterior, workflow_manager: Manager, run_type : str, target):
    configuration = workflow_manager.configuration

    dim = fm.dim

    # initial training set
    train_p = training_set["train_p"]
    train_y = training_set["train_y"]
    errors = training_set["errors"]

    # active learning parameters
    training_config = configuration["training_config"]
    sampling_config = configuration["sampling_config"]


    sample_every = sampling_config["sample_every"]

    points_per_it = sample_every * training_config["points_per_it"]
    n_it = training_config["n_it"]//sample_every

    default_tol = training_config["default_tol_fixed"]

    FE_cost = configuration["forward_model_config"]["FE_cost"]

    n_init_samples = sampling_config["init_samples"]
    n_final_samples = sampling_config["final_samples"]
    n_walkers = sampling_config["n_walkers"]
    samples = np.array([np.zeros(dim)])
    

    for i in range(n_it) :

        print("Sampling posterior...") 
        print()

        n_burn = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it)**2 )
        n_samples = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it))
        # sample posterior
        new_samples = posterior.sample_points(n_samples)

        # update sample chains
        if n_burn * n_walkers > len(samples) :
            samples = np.array([np.zeros(dim)])
        else :
            samples = samples[n_burn*n_walkers :]
        samples = np.concatenate( (samples, new_samples), axis = 0)
        print("Done.")
        print()
        shortened_samples = samples[::5]

        # monitor convergence
        W = np.sum(errors**(-FE_cost))

        curr_target = target(surrogate, shortened_samples)

        # save results
        workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
        training_set["train_p"] = train_p
        training_set["train_y"] = train_y
        training_set["errors"] = errors
        workflow_manager.state_saver(run_type, i, W, training_set, surrogate, samples)

        print("Rietriving candidates...")
        print()
        # position problem
        indices = np.random.randint(0, len(new_samples), size = points_per_it)
        new_pts = new_samples[indices]
        print("Done.")
        print()

        new_tols = default_tol * np.ones(len(new_pts))

        print("Evaluating model...")
        print()
        # evaluate model
        new_vals, new_errs = fm.predict(new_pts, new_tols)
        
        print("Done.")
        print()

        print("Updating surrogate...")
        print()
        # update surrogate
        train_p =  np.concatenate((train_p, new_pts), axis = 0)
        train_y = np.concatenate((train_y, new_vals), axis = 0)
        errors = np.concatenate((errors, new_errs), axis = 0)

        surrogate.fit(train_p, train_y, errors**2)
        print("Done.")
        print()

        new_target = target(surrogate, shortened_samples)


        print()
        print(f"Iteration {i}")
        print(f"Precedent target approx value: {curr_target}")
        print(f"Current target approx value: {new_target}")
        print(f"Points in the training set: {len(train_p)}")
        print()

        gc.collect()
    
    # export final round, save
    n_burn = n_walkers * n_samples
    n_samples = n_final_samples
    # sample posterior
    new_samples = posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

    shortened_samples = samples[::5]

    # monitor convergence
    W = np.sum(errors**(-FE_cost))

    curr_target = target(surrogate, shortened_samples)

    # save results
    workflow_manager.save_results({"W": [W], "target": [curr_target]}, run_type)
    workflow_manager.state_saver(run_type, n_it, W, training_set, surrogate, samples)
    

def run_adaptive_test(training_set : dict, surrogate : Surrogate, fm : forward_model, posterior : Posterior, workflow_manager: Manager, true_posterior : Posterior = None, ):
    configuration = workflow_manager.configuration

    param_space = fm.dom
    dim = fm.dim

    # initial training set
    train_p = training_set["train_p"]
    train_y = training_set["train_y"]
    errors = training_set["errors"]

    # active learning parameters
    training_config = configuration["training_config"]
    sampling_config = configuration["sampling_config"]

    points_per_it = training_config["points_per_it"]
    n_it = training_config["n_it"]
    sample_every = sampling_config["sample_every"]

    default_tol = training_config["default_tol_ada"]
    budget_ada = training_config["budget"]
    budget_per_it = budget_ada / n_it

    FE_cost = configuration["forward_model_config"]["FE_cost"]

    n_init_samples = sampling_config["init_samples"]
    n_final_samples = sampling_config["final_samples"]
    n_walkers = sampling_config["n_walkers"]
    samples = np.array([np.zeros(dim)])
    

    true_samples = true_posterior.sample_points(4000)

    tr_mean = np.mean(true_samples, axis = 0)
    tr_std  = np.sqrt( np.mean(true_samples**2, axis = 0 )- tr_mean**2)
    cleaned_true = true_samples[np.all( true_samples - tr_mean < 4*tr_std, axis = 1)]

    for i in range(n_it) :
        if i%sample_every == 0 :
            print("Sampling posterior...") 
            print()

            n_burn = int(n_init_samples + (n_final_samples -  n_init_samples)* (i/n_it)**2 )
            n_samples = int(n_init_samples  + (n_final_samples -  n_init_samples)* (i/n_it))
            # sample posterior
            new_samples = posterior.sample_points(n_samples)

            # update sample chains
            if n_burn * n_walkers > len(samples) :
                samples = np.array([np.zeros(dim)])
            else :
                samples = samples[n_burn*n_walkers :]
            samples = np.concatenate( (samples, new_samples), axis = 0)

            print(f'n_samples: {32 * n_samples} n_burn: {32 * n_burn}')
            print(f"Samples: {len(samples)}")
            print(f"new_samples : {len(new_samples)}")
            print("Done.")
            print()
            shortened_samples = samples[::5]

        samp_mean = np.mean(samples, axis = 0)
        samp_std  = np.sqrt( np.mean(samples**2, axis = 0 )- samp_mean**2)
        cleaned_samples = samples[np.all( samples - samp_mean < 4*samp_std, axis = 1)]

        corner_plot(
            [cleaned_samples, cleaned_true[:len(cleaned_samples)]], 
            colors = ["teal", "crimson"],
            labels = ["Adaptive approximation", "Ground truth"],
            points = [train_p],
            points_colors = ["black"],
            title = f"Samples at iteration {i}",
            domain = param_space,
            savepath = configuration["res_path"] + f"/samples_{i}.png",
        )
        

        _, std = surrogate.predict(shortened_samples, return_std=True)

        curr_L2 = AL.L2_GP.L2_approx(std**2).mean()

        print("Rietriving candidates...")
        print()
        # position problem
        candidates = solve_pos_prob(points_per_it, param_space, default_tol, surrogate, shortened_samples,  FE_cost )
        print("Done.")
        print()

        print("Optimizing tolerances...")
        print()
        # accuracy problem
        tolerances, new_pts, updated = solve_acc_prob(candidates, budget_per_it, surrogate, shortened_samples, FE_cost)
        print("Done.")
        print()

        new_tols = tolerances[len(train_p):]
        update_tols = tolerances[:len(train_p)]

        print("Evaluating model...")
        print()
        # evaluate model

        if np.any(updated) :
            train_y[updated], errors[updated] = fm.predict(train_p[updated], update_tols[updated])

        if len(new_pts) > 0:
            new_vals, new_errs = fm.predict(new_pts, new_tols)

            train_p =  np.concatenate((train_p, new_pts), axis = 0)
            train_y = np.concatenate((train_y, new_vals), axis = 0)
            errors = np.concatenate((errors, new_errs), axis = 0)
        
        print("Done.")
        print()

        print("Updating surrogate...")
        print()

        # update surrogate
        surrogate.fit(train_p, train_y, errors**2)
        print("Done.")
        print()


        # monitor convergence
        # save results
        W = np.sum(errors**(-FE_cost))

        _, std = surrogate.predict(samples, return_std=True)
        new_L2 = AL.L2_GP.L2_approx(std**2).mean()

        print()
        print(f"Iteration {i}")
        print(f"Precedent L2 approx value: {curr_L2}")
        print(f"Current L2 approx value: {new_L2}")
        print(f"Points in the training set: {len(train_p)}")
        print()

        gc.collect()
    
    # export final round, save
    n_burn = n_walkers * n_samples
    n_samples = n_final_samples
    # sample posterior
    new_samples = posterior.sample_points(n_samples)

    # update sample chains
    samples = samples[n_burn :]
    samples = np.concatenate( (samples, new_samples), axis = 0)

    samp_mean = np.mean(samples, axis = 0)
    samp_std  = np.sqrt( np.mean(samples**2, axis = 0 )- samp_mean**2)
    cleaned_samples = samples[np.all( samples - samp_mean < 4*samp_std, axis = 1)]

    shortened_samples = samples[::5]

    # monitor convergence
    W = np.sum(errors**(-FE_cost))

    _, std = surrogate.predict(shortened_samples, return_std=True)
    curr_L2 = AL.L2_GP.L2_approx(std**2).mean()

    corner_plot(
        [cleaned_samples, cleaned_true[:len(cleaned_samples)]], 
        colors = ["teal"],
        labels = ["Ground truth", "AGP approximation"],
        points = [train_p],
        points_colors = ["black"],
        title = f"Samples at iteration {i+1}",
        domain = param_space,
        savepath = configuration["res_path"] + f"/samples_{i+1}.png",
    )
