#%%
import enum
import pandas as pd
import numpy as np
import math
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, roc_curve
from evaluation_tool.utils import round_05

from src.evaluation_tool.layered_tool import FairKit

#%% Main
if __name__ == "__main__":
    compas_file_path = 'data\\processed\\compas\\compas-scores-two-years-pred.csv'
    compas = pd.read_csv(compas_file_path)
    compas.head()

    fair_compas = FairKit(
        y = compas.two_year_recid, 
        y_hat = compas.pred_medium_high, 
        a = compas.age_cat, 
        r = compas.decile_score,
        model_type='COMPAS Decile Scores')
    #fair_compas.l2_plot()


    # Plotting spider plot 

    # Extracting rates for each group 
    discrim_rates = ['FPR', 'FNR', 'FDR', 'FOR']
    name_map = {}
    rel_rates = pd.DataFrame({'rate': discrim_rates})
    for i, grp in enumerate(fair_compas.sens_grps):
        grp_lab = 'grp' + str(i)
        name_map[grp_lab] = grp
        rel_rates[grp_lab] = [
            fair_compas.rates[grp][rate] for rate in rel_rates.rate
            ]

    # ratio of each group's rate vs. minimum rate across groups 
    for grp in name_map.keys():
        pair_name = grp + "_vs_min_rate_ratio"
        assign_dict = {pair_name: lambda x: (x[grp] - x[list(name_map.keys())].min(axis = 1))/abs(x[list(name_map.keys())].min(axis = 1))*100}
        rel_rates = rel_rates.assign(**assign_dict)
    rel_rates = rel_rates.assign(grp_w_min_rate = lambda x: x[list(name_map.keys())].idxmin(axis = 1))
    rel_rates['grp_w_min_rate_name'] = [name_map[i] for i in rel_rates.grp_w_min_rate]

    fig = go.Figure() 
    for i in [0, 2, 3]:
        fig.add_trace(go.Scatterpolar(
        r=list(rel_rates.filter(regex= 'vs').iloc[i]),
        theta=fair_compas.sens_grps,
        fill='toself',
        name = rel_rates.rate[i]
        ))
    
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True
        ),
    ),
    showlegend=True
    )

    fig.show()



    # spider plot with matplotlib 

    # Colors of rates
    palette = list(sns.color_palette('Paired').as_hex()) 
    rate_cols = [palette[i] for i in [2,3,8,9]]
    
    rates = rel_rates.rate.tolist() # Each line in plot 
    groups = list(name_map.keys()) # Each corner in plot 
    n_groups = len(groups)

    # Caculating angles to place the x at
    angles_x = [n/n_groups*2*np.pi for n in range(n_groups)]
    angles_x += angles_x[:1]

    angles_all = np.linspace(0, 2*np.pi)
#%%
    as_subplots = True 
    fig = plt.figure(figsize = (10,8))

    if as_subplots == False:
        ax = fig.add_subplot(1, 1, 1, polar = True)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)

    for i, rate in enumerate(rel_rates.rate.tolist()):
        if as_subplots:
            ax = fig.add_subplot(2, 2, i+1, polar = True)
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
        
        rate_vals = rel_rates.filter(regex= 'vs').iloc[i].tolist()
        rate_vals += rate_vals[:1]
        ax.plot(angles_x, rate_vals, c = rate_cols[i], linewidth = 4, label = rate)
        ax.fill(angles_x, rate_vals, rate_cols[i], alpha=0.5)
        tol = 5
        ax.fill(angles_all, np.ones(len(angles_all))*tol, "#2a475e", alpha = 0.7)

        if as_subplots:
            # putting x labels as sensitive groups 
            ax.set_xticks(angles_x[:-1])
            ax.set_xticklabels(list(name_map.values()), size = 14)
            # y tick and  labels 
            yticks = [round_05(np.linspace(0, max(rate_vals), 5)[i]) for i in range(5)]
            ytick_labels = [f"{int(ax.get_yticks()[i])}%" for i in range(len(ax.get_yticks()))]
            ytick_labels[0] = ''
            print(yticks)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{int(ax.get_yticks()[i])}%" for i in range(len(ax.get_yticks()))])
            # Remove spines
            ax.spines["polar"].set_color("none")
            legend = ax.legend(loc=(1, 0),       # bottom-right
                        frameon=False     # don't put a frame
                )

    # putting x labels as sensitive groups 
    ax.set_xticks(angles_x[:-1])
    ax.set_xticklabels(list(name_map.values()), size = 14)
    ax.set_yticklabels([f"{int(ax.get_yticks()[i])}%" for i in range(len(ax.get_yticks()))])
    # Remove spines
    ax.spines["polar"].set_color("none")
    # Adding tolerance 
    tol = 5
    ax.fill(angles_all, np.ones(len(angles_all))*5, "#2a475e")
    legend = ax.legend(loc=(1, 0),       # bottom-right
                       frameon=False     # don't put a frame
             )

# %%

# %%
