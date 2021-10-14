#%% Imports
import pandas as pd
import numpy as np
import math

# Widgets
from ipywidgets import interactive, fixed, interact, FloatSlider
from IPython.display import display

# Plots 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

import seaborn as sns
from seaborn.palettes import color_palette

# sklearn 
from sklearn.metrics import confusion_matrix, roc_curve

# dir functions
from src.evaluation_tool.utils import (
    cm_matrix_to_dict, custom_palette, abs_percentage_tick, flatten_list, cm_dict_to_matrix, add_colors_with_stripes, get_alpha_weights,
    value_counts_df, desaturate, label_case, format_text_layer_1)

#%%

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

    # normalization constant
    if w_fp < 0.5:
        c = 1/(1-w_fp)
    else:
        c = 1/w_fp

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
            r (float array): Scores of length n. The threshold is assumed to be 0.5. 
            w_fp (int or float): False positive error rate
            model_type (str): Name of the model or dataset used.
        
        """
        
        self.y = y
        self.y_hat = y_hat
        self.a = a
        self.r = r
        self.model_type = model_type
        self.w_fp = w_fp

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'y_hat': y_hat, 'r': r})
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
    #                  LAYER METHODS
    ###############################################################
    
    def layer_1(self, plot  = True, output_table = True, w_fp = None):
        """To do: Documentation"""
        if w_fp is None:
            w_fp = self.w_fp

        wmr = self.get_WMR_rates(w_fp = w_fp)
        relative_wmr = self.get_relative_rates(wmr)
        obs_counts = (value_counts_df(self.classifier, 'a')
            .rename(columns = {'a': 'grp'}))
        l1_data = (pd.merge(relative_wmr, obs_counts)
            .rename(columns = {
                'relative_rate': 'weighted_misclassification_ratio'}))
        l1_data = l1_data[['grp', 'n', 'weighted_misclassification_ratio']]

        if plot:
            self.plot_layer_1(l1_data=l1_data, ax = None)
            #print('Whoops, we still need to implement l1 plot')

        if output_table:
            return l1_data
    
    def layer_2(self, plot = True, output_table = True, w_fp = None):
        """To do: Documentation"""
        if w_fp is None:
            w_fp = self.w_fp
    
        if plot:
            gs = GridSpec(nrows = 10, ncols = 3)
            f = plt.figure(figsize=(20,6))
            ax0 = f.add_subplot(gs[:, 0])
            ax2 = f.add_subplot(gs[5:,1:2])
            ax1 = f.add_subplot(gs[0:4,1:2], sharex = ax2)

            self.plot_rates(ax = ax0, w_fp = w_fp)
            self.plot_relative_rates(ax = ax1, w_fp = w_fp)
            self.plot_fairness_barometer(ax = ax2, w_fp = w_fp)

            f.subplots_adjust(wspace = 0.5, hspace = 0.7)

        if output_table:
            rates = pd.concat(
                [self.rates, self.get_WMR_rates(w_fp = w_fp)]
                ).reset_index(drop = True)
            relative_rates = self.get_relative_rates(rates = rates)
            barometer = self.get_fairness_barometer(w_fp = w_fp)
            return rates, relative_rates, barometer
        
    def layer_3(self, method, plot = True, output_table = True, w_fp = None):
        """To do: Documentation"""

        # To do: Split up in getting data and getting plot
        if w_fp is None:
            w_fp = self.w_fp

        method_options = [
            'w_fp_influence', 
            'roc_curves', 
            'calibration', 
            'confusion_matrix']

        if not isinstance(method, str):
            raise ValueError(f'`method` must be of type string. You supplied {type(method)}')
        
        if method not in method_options:
            raise ValueError(f'`method` must be one of the following: {method_options}. You supplied `method` = {method}')

        if method == 'w_fp_influence':
            # To do: Get data out?
            self.plot_w_fp_influence()

        if method == 'roc_curves':
            # To do: Fix that get_roc_curves is called twice
            roc = self.get_roc_curves()
            if plot:
                self.plot_roc_curves()
            if output_table:
                return roc

        if method == 'calibration':
            pass

        if method == 'confusion_matrix':
            # To do: Get data out in a sensible way
            self.plot_confusion_matrix()
        

        # To do: Make this :)

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

    def get_relative_rates(self, rates = None, rate_names = None):
        """Calculate relative difference in rates between group rate 
        and minimum rate.

        Args:
            rates (DataFrame): Contains data frame with rates from which relative rates should be calculated
            rate_names (list): list of names of rates for which to calculate relative rates
        """
        def get_minimum_rate(group):
            group['min_rate'] = group['rate_val'].agg('min')
            return group
        
        if rate_names == [np.nan]:
            return None
        elif rate_names is not None:
            rates = self.rates[self.rates.rate.isin(rate_names)]
        elif rates is None and rate_names is None:
            rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
            rates = self.rates[self.rates.rate.isin(rate_names)]
    

        # Calculate relative rates
        rel_rates = (rates
            .groupby(by = 'rate')
            .apply(get_minimum_rate)
            .assign(
                relative_rate = lambda x: 
                    (x.rate_val-x.min_rate)/x.min_rate*100)
            .drop(columns = ['min_rate']))

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

        # Decide unfavorable outcome used for independence measure
        if w_fp > 0.5:
            independence_measure = 'PP/n'
        elif w_fp < 0.5:
            independence_measure = 'PN/n'
        else:
            # Independence not measured if w=0.5
            independence_measure = np.nan


        fairness_crit = pd.DataFrame([
            ['Independence', independence_measure],
            ['Separation', 'FPR'],
            ['Separation', 'FNR'],
            ['FPR balance', 'FPR'],
            ['Equal opportunity', 'FNR'],
            ['Sufficiency', 'FDR'],
            ['Sufficiency', 'FOR'],
            ['Predictive parity', 'FDR'],
            ['WMR balance', 'WMR']],
            columns = ['criterion', 'rate']).dropna()
        
        rel_WMR = self.get_relative_WMR(w_fp = w_fp)
        rel_independence = self.get_relative_rates(
            rate_names = [independence_measure])
        all_data = (pd.concat([rel_WMR, self.rel_rates, rel_independence])
            .merge(fairness_crit))
        idx_discrim = (all_data
            .groupby(by = ['rate', 'criterion'], as_index = False)
            .relative_rate
            .idxmax()) # Idx for discriminated groups
        fairness_barometer = (all_data.loc[idx_discrim.relative_rate]
            .groupby(by = 'criterion', as_index = False)
            .agg({
                'relative_rate': 'mean',
                'grp': lambda x: list(pd.unique(x))})
            .rename(columns = {'grp': 'discriminated_grp'})
            .sort_values('relative_rate', ascending = False))
        return fairness_barometer

    def get_roc_curves(self):
        # To do: Documentation
        roc_list = []
        for grp in self.sens_grps:
            data_grp = self.classifier[self.classifier.a == grp]
            fpr, tpr, thresholds = roc_curve(
                y_true = data_grp.y, 
                y_score = data_grp.r)
            roc_list.append(pd.DataFrame({
                'fpr': fpr, 
                'tpr': tpr, 
                'threshold': thresholds,
                'sens_grp': grp}))
        roc = pd.concat(roc_list).reset_index(drop = True)  
        return roc  

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

    def plot_rates(self, ax = None, w_fp = None):
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
        containers = flatten_list(ax.containers) # order = from left to right by group
        for bar, rate in zip(containers, rate_names*self.n_sens_grps):
            alpha = alpha_weights[rate]
            bar.set_alpha(alpha)

        return ax

    def plot_relative_rates(self, ax = None, w_fp = None):
        """Plot the rate ratio for each sensitive groups
        
        Args:
            w_fp (float): False positive error weight
        
        """
        if w_fp is None:
            w_fp = self.w_fp
        rate_positions = {'WMR': 1, 'FPR': 0.8, 'FNR': 0.6, 'FDR': 0.4, 'FOR': 0.2}
        alpha_weights = get_alpha_weights(w_fp)
        plot_df = (pd.concat([
                self.rel_rates, 
                self.get_relative_WMR(w_fp = w_fp)
            ])
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
        
    def plot_fairness_barometer(self, w_fp = None, ax = None):
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
            color_variable = plot_df.discriminated_grp)

    def l2_interactive_plot(self):
       return interact(self.l2_plot, w_fp=FloatSlider(min=0,max=1,atep=0.1,value=0.8))

    def plot_w_fp_influence(self, relative = True):
        """Visualize how w_fp influences the weighted misclassification ratio
        
        Args:
            relative (bool): Plot weighted misclassification ratio? If False, weighted misclassification rate is plotted
        """
        plot_df = (pd.concat(
                [self.get_relative_WMR(w_fp = w_fp).assign(w_fp = w_fp) 
                for w_fp in np.linspace(0, 1, num = 100)])
            .reset_index()
            .rename(columns = {
                'rate_val': 'weighted_misclassification_rate',
                'relative_rate': 'weighted_misclassification_ratio'}))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if relative:
            y = 'weighted_misclassification_ratio'
        else:
            y = 'weighted_misclassification_rate'
        sns.lineplot(
            x = 'w_fp', y = y, hue = 'grp', 
            data = plot_df, 
            ax = ax, 
            palette = fair_anym.sens_grps_cols)
        ax.axhline(y = 20, color = 'grey', linewidth = 0.5)
        sns.despine(ax = ax, top = True, right = True)
        ax.set_xlabel('$w_{fp}$')
        ax.set_ylabel(label_case(y))
        ax.legend(frameon = False)
        if relative:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
   
    def plot_layer_1(self, l1_data, ax = None):
        """ Visualize the maximum gap in WMR by text
        
        Args:
            l1_data (data frame): data frame with data returned in layer_1
        """
        if ax is None:
            fig = plt.figure(figsize=(6,2))
            ax = fig.add_subplot(1, 1, 1)

        p_grey = desaturate((58/255, 58/255, 58/255))
        max_idx = l1_data.weighted_misclassification_ratio.idxmax() 
        max_grp = l1_data.grp[max_idx]
        max_val = l1_data.weighted_misclassification_ratio[max_idx]
        max_color = desaturate(self.sens_grps_cols[max_grp])

        # Creating text lines
        line_1 = (f"The WMR of sensitive group").split()
        line_2 = (f"is {max_val:.0f}% larger than the minimum WMR").split()

        # customizing lines with group
        line_1 = line_1 + [f"'{max_grp.capitalize()}'"]
        n_words_1 = len(line_1)
        color_list_1 = [p_grey]*n_words_1
        font_sizes_1 = [20]*n_words_1
        font_weights_1 = ['normal']*n_words_1
        color_list_1[-1] = max_color # coloring group
        font_sizes_1[-1] = 30 # making group bigger
        font_weights_1[-1] = 'bold' 

        # Costumizing lines with max_val 
        n_words_2 = len(line_2)
        color_list_2 = [p_grey]*n_words_2
        font_sizes_2 = [20]*n_words_2
        font_weights_2 = ['normal']*n_words_2
        font_weights_2[1:2] = ['bold', 'bold'] 
        font_sizes_2[1:2] = [30, 30]

        # Plotting text on axis
        ax.set_xlim(0,1.1)
        ax.set_ylim(0.68,0.9)
        ax.set_axis_off()
        sns.despine(top = True, bottom = True, left = True, right= True)
        format_text_layer_1(ax, 0.02, 0.8, line_1, color_list_1,
                            font_sizes_1, font_weights_1)
        format_text_layer_1(ax, 0.02, 0.7, line_2, color_list_2,
                            font_sizes_2, font_weights_2)

    def plot_roc_curves(self):
        # To do: Documentation
        roc = self.get_roc_curves()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        sns.lineplot(
            x = 'fpr', y = 'tpr', hue = 'sens_grp', 
            data = roc, ax = ax,
            estimator = None, 
            palette = fair_anym.sens_grps_cols)
        sns.scatterplot(
            x = 'fpr', y = 'tpr', 
            data = chosen_threshold, ax = ax,
            marker = 'x', s = 100, color = 'black')
        ax.plot([0,1], [0,1], color = 'grey', linewidth = 0.5)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        sns.despine(ax = ax, top = True, right = True)
        ax.legend(frameon = False, loc = 'lower right')

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

    # l1 check
    l1 = fair_anym.layer_1()

    # l2 check
    l2_rates, l2_relative_rates, l2_barometer = fair_anym.layer_2()

    # l3 check
    fair_anym.layer_3(method = 'w_fp_influence')
    fair_anym.layer_3(method = 'confusion_matrix')
    fair_anym.layer_3(method = 'roc_curves')

# %%
