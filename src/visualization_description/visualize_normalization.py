#%%
import numpy as np
from ipywidgets import interactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec

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

    # Visualize weighted misclassification rate
    WMR_vals = get_WMR(FP_vals, FN_vals, N, w_fp)
    #fig, ax = plt.subplots()
    ax.set_xlabel('FP')
    ax.set_ylabel('FN')
    ax.set_title('$w_{fp}$ = '+ f'{w_fp}')
    sns.despine(ax = ax, top = True, right = True)
    im = ax.imshow(WMR_vals, origin = 'lower', cmap = 'BuPu')
    if col_bar:
        fig.colorbar(im, orientation='vertical')
    ax.show()


# %%
viz_wfp(0.5)
fig = plt.figure()
gs = GridSpec(nrows = 1, ncols = 3)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[2,0])

# %%
