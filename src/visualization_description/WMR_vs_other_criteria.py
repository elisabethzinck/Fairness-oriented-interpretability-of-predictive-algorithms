#%%
import pandas as pd
import numpy as np
from math import floor
from src.visualization_description.visualize_WMR_normalization_constant import wfp_normalization, get_WMR

from ipywidgets import interactive, fixed

import matplotlib.pyplot as plt
import seaborn as sns

def flatten_recursively(d):
    "From here: https://stackoverflow.com/questions/49773008/python-unlist-a-nested-element-in-a-list"
    v = [[i] if not isinstance(i, (list, np.ndarray)) else flatten_recursively(i) for i in d]
    return [i for b in v for i in b]

def construct_examples_grid(fraction_list = [0., 0.25, 0.5, 1], n = 100):
    n = 100
    P_frac, FP_frac, FN_frac = np.meshgrid(fraction_list, fraction_list, fraction_list)
    n_experiments = len(fraction_list)**3
    df = pd.DataFrame({
            'P_frac': flatten_recursively(P_frac),
            'FP_frac': flatten_recursively(FP_frac),
            'FN_frac': flatten_recursively(FN_frac)})

    df['P'] = n*df.P_frac
    df['N'] = n-df.P
    df['FP'] = df.FP_frac*df.N 
    df['FN'] = df.FN_frac*df.P
    df['FPR'] = df.FP/df.N
    df['FNR'] = df.FN/df.P

    return df

def duplicates_in_df(df, x_var, y_var):
    duplicates = df[[x_var, y_var]].duplicated()
    return duplicates.sum()>0

def make_heatmap_plot(plot_df, x_var, y_var, hue_var, ax = None):
    plot_df = plot_df[[x_var, y_var, hue_var]].drop_duplicates()
    if duplicates_in_df(plot_df, x_var, y_var):
        raise ValueError('Duplicates on x and y-axis not allowed in plot!')

    # Prepare palette and legend
    palette = sns.cubehelix_palette(start=.47, rot=-0.5, as_cmap=True)
    norm = plt.Normalize(plot_df[hue_var].min(), plot_df[hue_var].max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Make plot
    if ax is None:
        fig, ax = plt.subplots(1)
    plot_matrix = plot_df.pivot(x_var, y_var, hue_var)
    sns.heatmap(plot_matrix, cmap = palette, vmin = 0, vmax = 1)
    ax.invert_yaxis()

    return ax

def make_separation_plot(examples, w_fp, P_frac):
    plot_df = examples
    plot_df['WMR'] = get_WMR(FP = df.FP, FN = df.FN, N = n, w_fp = w_fp)

    x_var = 'FPR'
    y_var = 'FNR'
    hue_var = 'WMR'
    plot_df['P_frac_diff'] = (plot_df.P_frac - P_frac).abs()
    plot_df = plot_df[plot_df.P_frac_diff == min(plot_df.P_frac_diff)]

    plot_df = plot_df[[x_var, y_var, hue_var]].drop_duplicates()
    if duplicates_in_df(plot_df, x_var, y_var):
        raise ValueError('Duplicates on x and y-axis not allowed in plot!')

    make_heatmap_plot(plot_df, x_var = x_var, y_var = y_var, hue_var = hue_var)
    ax.set_title(f'WMR at w_fp = {w_fp} and P/n = {P_frac}')

    plt.show()

#%% Create samples
n = 100
step_size = 0.025
fraction_list = np.arange(start = 0, stop = 1, step = step_size)
examples = construct_examples_grid(n = 100, fraction_list = fraction_list)


#%% Play around with possible plots
df = examples
w_fp = 0.2
P_frac = 0.5
df[f'WMR({w_fp})'] = get_WMR(FP = df.FP, FN = df.FN, N = n, w_fp = w_fp)
plot_df = df
x_var = 'FPR'
y_var = 'FNR'
hue_var = f'WMR({w_fp})'
plot_df['P_frac_diff'] = (plot_df.P_frac - P_frac).abs()
plot_df = plot_df[plot_df.P_frac_diff == min(plot_df.P_frac_diff)]
ax = make_heatmap_plot(plot_df, x_var, y_var, hue_var)

#%% Separation plot
make_separation_plot(examples = df, w_fp = 0.3, P_frac = 0.6)

#%% Interactive separation plot
interactive(
    make_separation_plot, 
    w_fp = (0,1, 0.1),
    P_frac = (0.05, 0.95, 0.05),
    examples = fixed(examples))

# %%


# %%
