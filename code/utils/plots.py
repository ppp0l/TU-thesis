import numpy as np

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
        plt.legend(handles=lines, loc='upper right', )
    if title is not None:
        plt.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, format=saveformat)
    
    plt.close()

    

