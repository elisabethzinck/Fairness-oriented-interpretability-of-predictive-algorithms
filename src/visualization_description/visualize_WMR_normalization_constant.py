#%%
import numpy as np
from ipywidgets import interactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec

fig_path_report = '../Thesis-report/00_figures/'
# %% Visualize how normalization works
def wfp_normalization(w_fp):
    if w_fp < 1/2:
        c = 1/(1-w_fp)
    else:
        c = 1/w_fp
    return c

def get_WMR(FP, FN, N, w_fp):
    normalization = wfp_normalization(w_fp)

    WMR = normalization*(w_fp*FP + (1-w_fp)*FN)/N

    return WMR

def viz_wfp(fig, ax, w_fp, col_bar = True):
    # Generate grid of possible values
    N = 1000
    possible_vals = np.linspace(0,N, num = N+1)
    FP_vals,FN_vals = np.meshgrid(possible_vals, possible_vals)

    # Remove obs where FP + FN > N
    idx_too_many_obs = (FP_vals + FN_vals) > N
    FP_vals[idx_too_many_obs] = np.nan
    FN_vals[idx_too_many_obs] = np.nan

    # Colormap
    cmap = sns.cubehelix_palette(start=.47, rot=-0.5, as_cmap=True)

    # Visualize weighted misclassification rate
    WMR_vals = get_WMR(FP_vals, FN_vals, N, w_fp)
    ax.set_xlabel('FP', fontsize = 12)
    ax.set_ylabel('FN', fontsize = 12)
    ax.set_title('$w_{FP}$ = '+ f'{w_fp}', fontsize = 13)
    sns.despine(ax = ax, top = True, right = True)
    im = ax.imshow(WMR_vals, origin = 'lower', cmap = cmap)

    # Ticks
    ticks = [0, N*0.25, N*0.5, N*0.75, N]
    tick_labels = ["0%","25%", "50%", "75%", "100%"]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    
    if col_bar:
        fig.colorbar(im, orientation='vertical')

    return im


if __name__ == "__main__":
    # Defining axes
    fig = plt.figure(figsize = (10,7))
    gs = GridSpec(nrows = 1, ncols = 3)

    weights = [0, 0.5, 1]

    for i, w_fp in enumerate(weights):
        ax = fig.add_subplot(gs[0,i])
        im = viz_wfp(fig = fig, ax = ax, w_fp = w_fp, col_bar=False)

    # adjustments 
    fig.subplots_adjust(wspace = 0.5, hspace = 0.7, right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.378, 0.03, 0.25])
    cbar_ax.set_title('$WMR$', fontsize = 14)
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(fig_path_report+'WMR_triangles.pdf', bbox_inches='tight')

# %%