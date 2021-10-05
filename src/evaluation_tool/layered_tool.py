#%% Imports
import pandas as pd
import numpy as np
import math

# Widgets
from ipywidgets import interactive, fixed, interact, FloatSlider
import ipywidgets as widgets
from IPython.display import display

# Plots 
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
from seaborn.palettes import color_palette

# sklearn 
from sklearn.metrics import confusion_matrix, roc_curve

# dir functions
from src.evaluation_tool.utils import (
    cm_matrix_to_dict, custom_palette, abs_percentage_tick, flatten_list, cm_dict_to_matrix, add_colors_with_stripes, get_alpha_weights)

#%%
def get_minimum_rate(group):
    """Helper function to calculate minimum rate by group"""
    group['min_rate'] = group['rate_val'].agg('min')
    return group


def calculate_WMR(cm, w_fp):
    """Calculate weighted misclassification rate
    
    Args:
        cm (dict): Confusion matrix dictionary containing keys 'FP' and 'FN'
        w_fp (int or float): False positive error weight 
    """

    # Input check of w_fp
    if not isinstance(w_fp, int) and not isinstance(w_fp, float):
        raise TypeError("w_fp must be a float or integer.")
    if w_fp < 0 or w_fp > 1:
        raise ValueError(f"w_fp must be in [0,1]. You supplied w_fp = {w_fp}")

    c = min(1/(1-w_fp), 1/w_fp) # normalization constant
    n = sum(cm.values())
    wmr = c*(w_fp*cm['FP'] + (1-w_fp)*cm['FN'])/n
    return wmr

class FairKit:
    def __init__(self, y, y_hat, a, r, w_fp, model_type = None):
        """Saves and calculates all necessary attributes for FairKit object
        
        Args:
            y (binary array): Targets for model of length n
            y_hat (binary array): Predictions of length n
            a (string array): Sensitive groups of length n. 
            r (float array): Scores of length n
            w_fp (int or float): False positive error rate
            model_type (str): Name of the model or dataset used.
        
        """
        
        self.y = y
        self.y_hat = y_hat
        self.a = a
        self.r = r
        self.model_type = model_type
        self.w_fp = w_fp

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'y_hat': y_hat})
        self.sens_grps = np.sort(self.a.unique())
        self.n_sens_grps = len(self.sens_grps)
        

        self.cm = self.get_confusion_matrix()
        self.rates = self.get_rates()
        self.WMR_rates = self.get_WMR_rates()
        self.WMR_rel_rates = self.get_relative_rates(self.WMR_rates)
        self.rel_rates = self.get_relative_rates()
        self.sens_grps_cols = dict(
            zip(self.sens_grps, custom_palette(n_colors = self.n_sens_grps))
            )


    ###############################################################
    #                  CALCULATION METHODS
    ###############################################################
    def get_confusion_matrix(self):
        """Calculate the confusion matrix for sensitive groups"""
        cm = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            cm_sklearn = confusion_matrix(
                y_true = df_group.y, 
                y_pred = df_group.y_hat)
            cm[grp] = cm_matrix_to_dict(cm_sklearn)
        return cm


    def get_rates(self, w_fp = None):
        """Calculate rates by sensitive group"""
        if w_fp is None:
            w_fp = self.w_fp
        rates = {}   
        for grp in self.sens_grps:
            TP, FN, FP, TN = self.cm[grp].values()
            rates[grp] = {
                'TPR': TP/(TP + FN), 
                'FNR': FN/(TP + FN), 
                'TNR': TN/(TN + FP),
                'FPR': FP/(TN + FP),
                'PPV': TP/(TP + FP),
                'FDR': FP/(TP + FP),
                'NPV': TN/(TN + FN),
                'FOR': FN/(TN + FN),
                'PN/n': (TN+FN)/(TP+FP+TN+FN),
                'PP/n': (TP+FP)/(TP+FP+TN+FN)
                }

        # Convert rate dict to data frame
        rates = pd.DataFrame(
            [(grp, rate, val) 
            for grp,grp_dict in rates.items() 
            for rate,val in grp_dict.items()], 
            columns = ['grp', 'rate', 'rate_val'])
        
        return rates

    def get_relative_rates(self, rates = None):
        """Calculate relative difference in rates between group rate 
        and minimum rate.

        Args:
            rate_names (DataFrame): Contains rates that should be calculated
        """
        if rates is None:
            rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
            rates = self.rates[self.rates.rate.isin(rate_names)]

        # Calculate relative rates
        rel_rates = (rates
            .groupby(by = 'rate')
            .apply(get_minimum_rate)
            .assign(
                relative_rate = lambda x: 
                    (x.rate_val-x.min_rate)/x.min_rate*100))

        return rel_rates
    
    def get_WMR_rates(self, w_fp = None):
        if w_fp is None:
            w_fp = self.w_fp
        WMR = pd.DataFrame({
            'grp': self.sens_grps,
            'rate': 'WMR'})
        WMR['rate_val'] = [calculate_WMR(self.cm[grp], w_fp) for grp in WMR.grp]
        return WMR

    def get_relative_WMR(self, w_fp = None):
        if w_fp == self.w_fp or w_fp is None:
            res = self.WMR_rel_rates
        else:
            WMR_rates = self.get_WMR_rates(w_fp)
            res = self.get_relative_rates(WMR_rates)
        return res

    def get_fairness_barometer(self, w_fp = None):
        if w_fp is None:
            w_fp = self.w_fp

        fairness_crit = pd.DataFrame([
            ['Independence', 'PN/n'],
            ['Separation', 'FPR'],
            ['Separation', 'FNR'],
            ['FPR balance', 'FPR'],
            ['Equal opportunity', 'FNR'],
            ['Sufficiency', 'FDR'],
            ['Sufficiency', 'FOR'],
            ['Predictive parity', 'FDR'],
            ['WMR balance', 'WMR']],
            columns = ['criterion', 'rate'])
        
        rel_WMR = self.get_relative_WMR(w_fp = w_fp)
        all_data = (pd.concat([rel_WMR, self.rel_rates])
            .merge(fairness_crit))
        idx = (all_data
            .groupby(by = ['rate', 'criterion'], as_index = False)
            .relative_rate
            .idxmax())
        fairness_barometer = (all_data.loc[idx.relative_rate]
            .groupby(by = 'criterion', as_index = False)
            .agg({
                'relative_rate': 'mean',
                'grp': lambda x: list(pd.unique(x))})
            .sort_values('relative_rate', ascending = False))
        return fairness_barometer


    ###############################################################
    #                  VISUALIZATION METHODS
    ###############################################################

    def plot_confusion_matrix(self):
        plt.figure(figsize = (15,5))
        n_grps = len(self.sens_grps)
        if self.model_type != None:
            plt.suptitle(self.model_type)

        # One plot for each sensitive group
        for i, grp in enumerate(self.sens_grps):
            n_obs = sum(self.cm[grp].values())
            grp_cm = cm_dict_to_matrix(self.cm[grp])
    
            plt.subplot(1,n_grps,i+1)
            ax = sns.heatmap(
                grp_cm/n_obs*100, 
                annot = True, 
                cmap = 'Blues', 
                vmin = 0, vmax = 100,
                cbar = False,
                xticklabels=['Predicted positive', 'Predicted negative'],
                yticklabels=['Actual positive', 'Actual negative'], 
                annot_kws={'size':15})

            # Adjust figure labels
            names = ['TP', 'FN', 'FP', 'TN']
            for name, a in zip(names, ax.texts): 
                old_text = a.get_text()
                new_text = f"{name}: {old_text}%"
                a.set_text(new_text)
            plt.ylabel(None)
            plt.xlabel(None)
            plt.title(f'{str.capitalize(grp)} (N = {n_obs})')


    def l2_rate_subplot(self, ax = None, w_fp = None):
        """Plot FPR, FNR, FDR, FOR for each group"""
        rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
        if w_fp is None:
            w_fp = self.w_fp
        plot_df = self.rates[self.rates.rate.isin(rate_names)]
        alpha_weights = get_alpha_weights(w_fp)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x = 'rate', y = 'rate_val', 
            hue = 'grp', palette = self.sens_grps_cols,
            data = plot_df,
            order = rate_names,
            ax = ax)
        ax.legend(loc = 'upper right', frameon = False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Group rates', fontsize=14, loc = 'left')
        ax.set_ylim(0,1)
        sns.despine(ax = ax, bottom = True, top = True, right = True)
        ax.tick_params(labelsize=12)

        # Set alpha values
        containers = flatten_list(
            [list(ax.containers[i][0:4]) for i in range(self.n_sens_grps)])
        for bar, rate in zip(containers, plot_df.rate):
            alpha = alpha_weights[rate]
            bar.set_alpha(alpha)

        return ax

    def l2_ratio_subplot(self, ax = None, w_fp = None):
        """Plot the rate ratio for each sensitive groups
        
        Args:
            w_fp (float in (0,1)): False positive error weight
        
        """
        if w_fp is None:
            w_fp = self.w_fp
        rate_positions = {'WMR': 1, 'FPR': 0.8, 'FNR': 0.6, 'FDR': 0.4, 'FOR': 0.2}
        alpha_weights = get_alpha_weights(w_fp)
        plot_df = (pd.concat([
                self.rel_rates, 
                self.get_relative_WMR(w_fp = w_fp)
            ])
            .query("rate != 'PN/n'")
            .assign(
                rate_position = lambda x: x.rate.map(rate_positions),
                alpha = lambda x: x.rate.map(alpha_weights)))
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.hlines(
            y= 'rate_position', xmin = 0, xmax = 'relative_rate',
            data = plot_df,
            color = 'lightgrey', linewidth = 1,
            zorder = 1)
        for _, alpha in enumerate(plot_df.alpha.unique()):
            sns.scatterplot(
                data = plot_df[plot_df['alpha'] == alpha], 
                x = 'relative_rate', y='rate_position', hue='grp',
                palette = self.sens_grps_cols, 
                legend = False,
                ax = ax,
                marker = 'o', alpha = alpha, s = 150,
                zorder = 2)
        ax.set_yticks(list(rate_positions.values()))
        ax.set_yticklabels(list(rate_positions.keys()))
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('Relative rates', fontsize=14, loc = 'left')
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(left = -0.05*xmax) # To see all of leftmost dots
        ax.set_ylim((.125,1.125))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
        sns.despine(ax = ax, left = True, top = True, right = True)
        ax.tick_params(left=False, labelsize=12)

    def l2_fairness_criteria_subplot(self, w_fp = None, ax = None):
        plot_df = self.get_fairness_barometer(w_fp = w_fp)
        if ax is None:
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x = 'relative_rate', y = 'criterion', 
            data = plot_df,
            ax = ax, zorder = 2)
        ax.axvline(x = 20, color = 'grey', zorder = 1, linewidth = 0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Unfairness barometer', fontsize=14, loc = 'left')
        sns.despine(ax = ax, left = True, top = True, right = True)
        ax.tick_params(left=False, labelsize=12)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
        add_colors_with_stripes(
            ax = ax, 
            color_dict = self.sens_grps_cols, 
            color_variable = plot_df.grp)
    

    def l2_plot(self, w_fp = None):
        
        # Define grid
        gs = GridSpec(nrows = 10, ncols = 3)
        f = plt.figure(figsize=(20,6))
        ax0 = f.add_subplot(gs[:, 0])
        ax2 = f.add_subplot(gs[5:,1:2])
        ax1 = f.add_subplot(gs[0:4,1:2], sharex = ax2)
        
        
        # Insert plots
        self.l2_rate_subplot(ax = ax0, w_fp = w_fp)
        self.l2_ratio_subplot(ax = ax1, w_fp = w_fp)
        self.l2_fairness_criteria_subplot(ax = ax2, w_fp = w_fp)

        # Adjustments
        f.subplots_adjust(wspace = 0.5, hspace = 0.7)

    def l2_interactive_plot(self):
       return interact(self.l2_plot, w_fp=FloatSlider(min=0,max=1,atep=0.1,value=0.8))
   

#%% Main
if __name__ == "__main__":
    file_path = 'data\\processed\\anonymous_data.csv'
    df = pd.read_csv(file_path)
    df.head()

    fair_anym = FairKit(
        y = df.y, 
        y_hat = df.yhat, 
        a = df.grp, 
        r = df.phat,
        w_fp = 0.8)
    fair_anym.l2_plot()
    #fair_anym.l2_fairness_criteria_subplot()
    #fair_anym.plot_confusion_matrix()
#%%

# %%
