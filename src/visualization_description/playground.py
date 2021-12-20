#%%
import numpy as np
from ipywidgets import interactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from biasbalancer.utils import (
    cm_matrix_to_dict, cm_dict_to_matrix, round_func)
from biasbalancer.BiasBalancerPlots import abs_percentage_tick
#%%
def plot_confusion_matrix(TP, FN, FP, TN):
        plt.figure(figsize = (15,5))
        cm = np.array([[TP, FN], [FP, FN]])
        n_obs = sum(sum(cm))
            
        plt.subplot(1,1,1)
        ax = sns.heatmap(
            cm/n_obs*100, 
            annot = True, 
            cmap = 'Blues', 
            vmin = 0, vmax = 100,
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'], 
            annot_kws={'size':15})
        for a in ax.texts: a.set_text(f"{a.get_text()}%")
        plt.ylabel('Actual (%)')
        plt.xlabel('Predicted (%)')
        plt.title(f'{n_obs} observations')
        plt.show()

#%%
def plot_playground(TP, FN, FP, TN, alpha):
        beta = 1-alpha
        n = TP + FN + FP + TN
        rates = {
            'FNR': FN/(TP + FN), 
            'FPR': FP/(TN + FP),
            'FDR': FP/(TP + FP),
            'FOR': FN/(TN + FN)}
        
        measures = {
            '(alpha*FP + beta*FN)/n': 
                (alpha*FP + beta*FN)/n,
            #'(alpha*FPR + beta*FNR)': 
            #    alpha*rates['FPR'] + beta*rates['FNR'],
            #'(alpha*FDR + beta*FOR)': 
            #    alpha*rates['FDR'] + beta*rates['FOR'],
            '(ny': 1/4*(alpha*rates['FPR'] + beta*rates['FNR'] + alpha*rates['FDR'] + beta*rates['FOR'])}

        measures.update(rates) # Add two dicts together

        measures_df = pd.DataFrame({
            'measure': measures.keys(),
            'val': measures.values()})


        #p9.ggplot(measures_df) + \
        #    p9.geom_col(p9.aes(x = 'measure', y = 'val')) + \
        #    p9.coord_flip()
        plt.barh(y = measures_df['measure'], width = measures_df['val'])
        plt.xlim(xmin = 0, xmax  = 1)
        plt.show()


#%%
def plot_playground_new(TP, FN, FP, TN, w_fp):
        w_fn = 1-w_fp
        n = TP + FN + FP + TN
        #rates = {
        #    'FNR': FN/(TP + FN), 
        #    'FPR': FP/(TN + FP),
        #    'FDR': FP/(TP + FP),
        #    'FOR': FN/(TN + FN)}
        
        measures = {
            'uden 2': 
                (w_fp*FP + w_fn*FN)/n,
            '2 i naevner':  (w_fp*FP + w_fn*FN)/(2*n),
            '2 i taeller':  2*(w_fp*FP + w_fn*FN)/n}

        #measures.update(rates) # Add two dicts together

        measures_df = pd.DataFrame({
            'measure': measures.keys(),
            'val': measures.values()})


        #p9.ggplot(measures_df) + \
        #    p9.geom_col(p9.aes(x = 'measure', y = 'val')) + \
        #    p9.coord_flip()
        plt.barh(y = measures_df['measure'], width = measures_df['val'])
        plt.xlim(xmin = 0, xmax  = 2)
        plt.show()
#%% Simple interactive plot with confusion matrix

interactive_plot = interactive(
    plot_confusion_matrix, 
    TP=(0,10),
    FP=(0,10),
    TN=(0,10),
    FN=(0,10))
interactive_plot

# %%
our_playground_new = interactive(
    plot_playground_new,
    TP=(0,100),
    FP=(0,100),
    TN=(0,100),
    FN=(0,100),
    w_fp = (0,1, 0.1))
our_playground_new

# %%
def check_degeneracy(TPa, FNa, FPa, TNa, TPb, FNb, FPb, TNb):
    cm = {}
    cm['a'] = cm_matrix_to_dict(np.array([[TPa, FNa], [FPa, TNa]]))
    cm['b'] = cm_matrix_to_dict(np.array([[TPb, FNb], [FPb, TNb]]))
    grps = ['a', 'b']
    rates = {}
    for grp in grps:
        TP, FN, FP, TN = cm[grp].values()
        rates[grp] = {
            'base_rate': (TP+FP)/(TP + FP + TN + FN), 
            'FNR': FN/(TP + FN), 
            'FPR': FP/(TN + FP),
            'FDR': FP/(TP + FP),
            'FOR': FN/(TN + FN)
            }

    rates_a = pd.DataFrame({
    'value': rates['a'].values(),
    'rate': rates['a'].keys(),
    'grp': 'a'})
    rates_b = pd.DataFrame({
    'value': rates['b'].values(),
    'rate': rates['b'].keys(),
    'grp': 'b'})
    rates = pd.concat([rates_a, rates_b])

    sns.barplot(x = 'value', y = 'rate', hue = 'grp', data = rates)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.show()
    return rates

# %%
our_playground = interactive(
    check_degeneracy,
    TPa=(0,2),
    FPa=(0,2),
    TNa=(0,2),
    FNa=(0,2),
    TPb=(0,2),
    FPb=(0,2),
    TNb=(0,2),
    FNb=(0,2))
our_playground
