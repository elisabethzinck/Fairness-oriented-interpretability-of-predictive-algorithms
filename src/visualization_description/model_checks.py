#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.evaluation_tool.layered_tool import FairKit
from src.visualization_description.evaluater import get_FairKitDict

#%%
def list_intersection(list1, list2):
    "Returns list of intersection between the lists"
    list1 = set(list1)
    list2 = set(list2)
    intersect = list(list1 & list2)
    return intersect

def plot_prediction_comparison(data, hue_var, title, hue_label = 'y_true'):
    "Plots comparison plot between logistic and nn predictions"
    ax = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(
        x = 'log_reg_prob', 
        y = 'nn_prob', 
        hue = hue_var,
        data = data,
        ax = ax,
        zorder = 2,
        alpha = 0.25)
    ax.axvline(x = 0.5, color = 'grey', zorder = 1, linewidth = 0.5)
    ax.axhline(y = 0.5, color = 'grey', zorder = 1, linewidth = 0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.legend(frameon = False, title = hue_label)
    sns.despine(ax = ax, top = True, right = True)
    ax.set_title(title)


#%% 
fairKitDict = get_FairKitDict()
# %% 
for dataset in ['german', 'catalan', 'taiwanese']:
    data_nn = fairKitDict[dataset+'_nn'].data
    data_logreg = fairKitDict[dataset+'_logreg'].data
    a_name = fairKitDict[dataset+'_nn'].a_name
    y_name = fairKitDict[dataset+'_nn'].y_name

    merge_cols = list_intersection(data_nn.columns, data_logreg.columns)
    data = data_nn.merge(data_logreg, on = merge_cols)
    plot_prediction_comparison(data, y_name, title = dataset)

# %%
