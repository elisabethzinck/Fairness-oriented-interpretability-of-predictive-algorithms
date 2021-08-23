#%%
import numpy as np
from ipywidgets import interactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
def plot_playground(TP, FN, FP, TN, alpha, beta):
        n = TP + FN + FP + TN
        rates = {
            'FNR': FN/(TP + FN), 
            'FPR': FP/(TN + FP),
            'FDR': FP/(TP + FP),
            'FOR': FN/(TN + FN)}
        
        measures = {
            '(alpha*FP + beta*FN)/n': 
                (alpha*FP + beta*FN)/n,
            '(alpha*FPR + beta*FNR)': 
                alpha*rates['FPR'] + beta*rates['FNR'],
            '(alpha*FDR + beta*FOR)': 
                alpha*rates['FDR'] + beta*rates['FOR']}

        measures.update(rates) # Add two dicts together

        measures_df = pd.DataFrame({
            'measure': measures.keys(),
            'val': measures.values()})


        #p9.ggplot(measures_df) + \
        #    p9.geom_col(p9.aes(x = 'measure', y = 'val')) + \
        #    p9.coord_flip()
        plt.barh(y = measures_df['measure'], width = measures_df['val'])
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
our_playground = interactive(
    plot_playground,
    TP=(0,10),
    FP=(0,10),
    TN=(0,10),
    FN=(0,10),
    alpha = (0,1, 0.1),
    beta = (0,1, 0.1))
our_playground

# %%
