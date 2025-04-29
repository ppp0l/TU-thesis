import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import corner

__all__ = ["corner_plot"]

def_colors = [
    "crimson",
    "teal",
    "gold",
    "darkviolet",
    "blue",
    "darkorange",
]
alt_colors = [
    "black",
    "brown",
    "deeppink",
    "forestgreen",
    "cyan"
    ]


def corner_plot(
        samples : list, 
        colors : list = def_colors,
        labels : list = None, 
        weights : list = None,
        domain :dict = None,
        points : list = [], 
        points_colors : list = alt_colors,
        title : str = "",
        savepath : str = None,
        saveformat : str = "png",
        **kwargs,
        ) :
        
    """
    Plot the corner plot of the samples and optionally points.
    """
    dim = len(samples[0][0])
    if weights is None:
        weights = [1 for _ in range(len(samples))]

    for i, chain in enumerate(samples):
        if i==0:
            fig = corner.corner(chain, color = colors[i], weights= weights[i] * np.ones(len(chain)), **kwargs)
        else:
            corner.corner(chain, fig=fig, color = colors[i], weights= weights[i] * np.ones(len(chain)), **kwargs)

    for i, points_set in enumerate(points):
        corner.overplot_points(fig,points_set, markersize = 2, color = points_colors[i], )
    
    if domain is not None:
        for i, ax in enumerate(fig.axes):
            l = i%dim
            j = i//dim
            if l <= j:
                ax.set_xlim(domain["min"][j], domain["max"][j])
                if l < j:
                    ax.set_ylim(domain["min"][l], domain["max"][l])

    if labels is not None:
        lines = [ mlines.Line2D([],[], color=colors[i], label=labels[i]) for i in range(len(labels)) ]
        plt.legend(handles=lines, loc = 7,bbox_to_anchor=(0., 1.9, 0.8, 1.0) )
    if title is not None:
        plt.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, format=saveformat)
    
    plt.close()


def plot_metric(path : str, type_surr : str) :

    path = path + '/outputs'
    picpath = path + "/pictures"

    types = ["A", "posAd", "LHS", "rand"]
    type_runs = [tipe + type_surr for tipe in types]

    colors = { run : def_colors[i] for i, run in enumerate(type_runs) }

    for subdir, dirs, files in os.walk(path):
        if not files :
            continue
        if subdir.split(os.sep)[-1].startswith(".") or subdir.split(os.sep)[0].startswith("."):
            continue

        whole_data = {run  : pd.DataFrame() for run in type_runs}
        start_ind = {run : [0] for run in type_runs}
        means = {run : [] for run in type_runs}
        
        found_csv = False
        for file in files:
            if not file.endswith('.csv') :
                continue
            
            #obtain info on data
            parsed = file.split('_')
            type_res = parsed[0]
            num_res = parsed[-1].split('.')[0]

            if type_res not in type_runs :
                continue

            found_csv = True
            filepath = subdir + os.sep + file
            
            #load data
            data = pd.read_csv(filepath)
            #fix type problem
            data.columns = data.columns.map(float)

            data = data.drop(-40, axis = 1, errors = 'ignore')
            
            #renumber data, save extrema to obtain means
            n_rows = data.index.stop
            old_end = start_ind[type_res][-1]
            start_ind[type_res].append(old_end + n_rows)
            new_index = pd.RangeIndex(start= old_end , stop= start_ind[type_res][-1], step=1)
            data.index = new_index
            
            #include in larger Dataframe
            whole_data[type_res] = pd.concat( (whole_data[type_res], data ) )
            
        if not found_csv : continue
        
        #minmax = min( [max(whole_data[key].columns) for key in whole_data.keys()]) +1
        
        for key in whole_data.keys() :
            #reshape data, interpolate missing values
            #whole_data[key][minmax] = np.nan
            whole_data[key] = whole_data[key].transpose().sort_index().interpolate(method='index')
            #whole_data[key] = whole_data[key].drop( whole_data[key][whole_data[key].index > minmax].index )
            
            #iterate to save means
            start = 0
            for end in start_ind[key] :
                if start==end :
                    continue
                rg = range(start, end)
                mean = whole_data[key][rg].mean(axis = 1)
                means[key].append(mean)
                start = end
        
        if type_surr == "GP" :
            ylabel = "Expected L2 error"
        else :
            ylabel = "Expected L1 error"
        
        dim = subdir.split(os.sep)[-1]
        #plot with colors
        plt.figure(figsize = (8, 3), layout='constrained')

        # total realizations
        for run in type_runs :
            n = len( whole_data[run].columns)
            for i in range(n):
                plt.semilogy( whole_data[run][i], colors[run], lw = 0.2, alpha = 0.1)

        for run in type_runs :
            for obj in means[run] :
                plt.semilogy(obj, colors[run], lw = 0.8, alpha = 0.5)

        for run in type_runs :
            plt.semilogy(whole_data[run].mean(axis = 1), colors[run], lw = 3, label = run)
        
        # print()
        # print( dim, fname)
        # print(f"geom mean : {norm_mean(whole_data['geom'])[minmax]}")
        # print(f"full mean : {norm_mean(whole_data['full'])[minmax]}")
        # print()
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('Computational work')


        currpicpath = picpath + os.sep + dim

        if not os.path.exists(currpicpath):
            os.makedirs(currpicpath)
        
        plt.savefig(currpicpath + os.sep + type_surr + "_res.png", format = 'png')
        # plt.savefig(currpicpath + os.sep + fname +".svg", format = 'svg')

        plt.savefig( subdir + os.sep + type_surr + "_results.png", format = 'png')
        # plt.savefig( subdir + os.sep + "results.svg", format = 'svg')
        

