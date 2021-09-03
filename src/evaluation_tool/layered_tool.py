#%%
from matplotlib import gridspec
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, roc_curve

from src.evaluation_tool.utils import (
    cm_matrix_to_dict, cm_dict_to_matrix, abs_percentage_tick, round_func)

#%%
class FairKit:
    def __init__(self, y, y_hat, a, r, model_type = None):
        """Saves and calculates all necessary attributes for FairKit object
        
        Args:
            y (binary array): Targets for model of length n
            y_hat (binary array): Predictions of length n
            a (string array): Sensitive groups of length n. 
            r (float array): Scores of length n
            model_type (str): Name of the model used. Defaults to None.
        
        """
        self.y = y
        self.y_hat = y_hat
        self.a = a
        self.r = r
        self.model_type = model_type

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'y_hat': y_hat})
        self.sens_grps = self.a.unique()
        self.n_sens_grps = len(self.sens_grps)
        self.w_fp = 0.5 

        self.get_confusion_matrix()
        self.get_rates()
        self.l2_get_relative_rates()

        # Define color palette
        n_grps = len(self.sens_grps)
        cols = sns.color_palette(n_colors = n_grps)
        self.sens_grps_cols = dict(zip(self.sens_grps, cols))
        self.sens_grps_no_cols = dict(zip(self.name_map.keys(), cols))

    def get_confusion_matrix(self):
        """Calculate the confusion matrix for sensitive groups"""
        self.cm = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            cm_sklearn = confusion_matrix(
                y_true = df_group.y, 
                y_pred = df_group.y_hat)
            self.cm[grp] = cm_matrix_to_dict(cm_sklearn)

    def get_rates(self):
        """Calculate rates by group"""
        self.rates = {}   
        for grp in self.sens_grps:
            TP, FN, FP, TN = self.cm[grp].values()
            self.rates[grp] = {
                'TPR': TP/(TP + FN), 
                'FNR': FN/(TP + FN), 
                'TNR': TN/(TN + FP),
                'FPR': FP/(TN + FP),
                'PPV': TP/(TP + FP),
                'FDR': FP/(TP + FP),
                'NPV': TN/(TN + FN),
                'FOR': FN/(TN + FN)
                }

    def l1_get_data(self, w_fp = 0.5):
        """Get data used for first layer of evaluation
        
        Args:
            w_fp (float): False positive error weight
        """
        self.w_fp = w_fp

        df = (pd.DataFrame(self.cm)
            .T.reset_index()
            .rename(columns = {'index':'grp'})
            .assign(
                n = lambda x: x.TP + x.FN + x.FP + x.TN,
                PP = lambda x: x.TP + x.FP,
                avg_w_error = lambda x: (w_fp*x.FP + (1-w_fp)*x.FN)/(2*x.n)))
        min_err = min(df.avg_w_error)
        df['perc_diff'] = (df.avg_w_error-min_err)/abs(min_err)*100

        return df

    def l1_plot(self):
        pass

    def l1_calculate(self):
        pass

    def l2_get_relative_rates(self):
        """Get relatice difference in rates between group rate 
        and minimum rate. Dataframe used for ratio plot in 
        second layer of evaluation
        """
        # Extracting rates for each group 
        discrim_rates = ['FPR', 'FNR', 'FDR', 'FOR']
        self.name_map = {}
        self.rel_rates = pd.DataFrame({'rate': discrim_rates})
        for i, grp in enumerate(self.sens_grps):
            grp_lab = 'grp' + str(i)
            self.name_map[grp_lab] = grp
            self.rel_rates[grp_lab] = [
                self.rates[grp][rate] for rate in self.rel_rates.rate
                ]

        # ratio of each group's rate vs. minimum rate across groups 
        grps = list(self.name_map.keys())
        for grp in grps:
            pair_name = grp + "_vs_min_rate_ratio"
            assign_dict = {
                pair_name: lambda x: (x[grp] - x[grps].min(axis = 1))/abs(x[grps].min(axis = 1))*100
                }
            self.rel_rates = self.rel_rates.assign(**assign_dict)

        self.rel_rates = self.rel_rates.drop(list(self.name_map.keys()), axis = 1)

        return self.rel_rates

    def l2_rate_subplot(self, axis = None):
        discrim_rates = ['FPR', 'FNR', 'FDR', 'FOR']
        rates_df = pd.DataFrame(
            [(grp, rate, val) 
            for grp,grp_dict in self.rates.items() 
            for rate,val in grp_dict.items()], 
            columns = ['grp', 'rate', 'val'])
        subset = rates_df[rates_df.rate.isin(discrim_rates)]

        if axis is None:
            axis = plt.gca()
        sns.barplot(
            x = 'val', y = 'rate', 
            hue = 'grp', palette = self.sens_grps_cols,
            data = subset,
            order = discrim_rates,
            ax = axis)
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.set_xlim(0,1)
        for pos in ['right', 'top', 'left']:
            axis.spines[pos].set_visible(False)
        axis.tick_params(left=False, labelsize=12)
        axis.legend(loc = 'upper right', frameon = False)

        return axis

    def l2_ratio_subplot(self, axis = None):
        
        if (self.rel_rates is None) | (self.name_map is None):
            self.l2_get_relative_rates()

        rel_rates_2grp_plot = self.rel_rates.assign(
            signed_ratio = lambda x: x["grp1_vs_min_rate_ratio"]-
            x["grp0_vs_min_rate_ratio"]
            )
        discrim_rates = list(rel_rates_2grp_plot.rate)

        if axis is None:
            axis = plt.gca()
        sns.barplot(
            x = 'signed_ratio', y = 'rate', 
            hue = 'max_grp_name', palette = self.sens_grps_cols,
            data = rel_rates_2grp_plot,
            order = discrim_rates,
            ax = axis,
            dodge = False)
        axis.axvline(x = 0, color = 'black')
        axis.set_xlabel('Percent increase in rate for group')
        axis.set_ylabel('')
        for pos in ['right', 'top', 'left']:
            axis.spines[pos].set_visible(False)
        axis.tick_params(left=False)
        axis.xaxis.set_major_formatter(
            mtick.FuncFormatter(abs_percentage_tick))
        axis.legend(title = None, frameon = False)

        return axis

    def l2_ratio_lollipop_subplot(self, ax = None):
        if (self.rel_rates is None) | (self.name_map is None):
            self.l2_get_relative_rates()

        w_size = 1000
        plot_df = self.rel_rates.assign(rate_vals = [1, 0.75, 0.5, 0.25], 
                              w = [w_size*self.w_fp, w_size*(1-self.w_fp), 
                                   w_size*(1-self.w_fp), w_size*self.w_fp]
                             )
        plot_df.columns = plot_df.columns.str.replace(r'_vs_min_rate_ratio','')
        rel_rates_tidy = plot_df.melt(id_vars = ['rate', 'rate_vals','w'], var_name='grp', value_name='vs_min_rate_ratio')

        if ax is None:
            ax = plt.gca()
        ax.hlines(y=rel_rates_tidy.rate_vals,
                xmin = 0,
                xmax = rel_rates_tidy.vs_min_rate_ratio,
                color = 'lightgrey', 
                alpha = 1, 
                linestyles='solid', 
                linewidth=1, 
                zorder = 1)
        sns.scatterplot(data = rel_rates_tidy, 
                        x = 'vs_min_rate_ratio',
                        y='rate_vals',
                        hue='grp',
                        palette = self.sens_grps_no_cols, 
                        size = 'w',
                        sizes = (100,200),
                        legend = False,
                        ax = ax,
                        marker = 'o',
                        alpha = 1, 
                        zorder = 2)
        ax.set_yticks(plot_df.rate_vals)
        ax.set_yticklabels(plot_df.rate)
        ax.set_ylabel('')
        ax.set_xlabel('Relative difference in Percent', fontsize=12)
        ax.set_title('Relative Difference of Group Rate vs. Minimum Rate', fontsize=14) 
        ax.set_ylim((.125,1.125))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
        for pos in ['right', 'top', 'left']:
                    ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, labelsize=12)
    
        return ax

    def l2_plot(self):
        if(len(self.sens_grps) <= 2):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            self.l2_ratio_subplot(axis = ax1)
            self.l2_rate_subplot(axis = ax2)
            f.subplots_adjust(wspace = 0.5, right = 1)
        else:
            gs = GridSpec(nrows = 3, ncols = 1)
            f = plt.figure(figsize=(7,6))
            ax_list = [f.add_subplot(gs[0:2,0]),
                       f.add_subplot(gs[2,0])]
            self.l2_ratio_lollipop_subplot(ax = ax_list[1])
            self.l2_rate_subplot(axis = ax_list[0])
            f.subplots_adjust(hspace = 0.8, right = 1)
        return f 

    def create_fake_example(self):
        """Modifies rate to show discrimination of women"""
        self.rates['female']['FPR'] = 0.47
        self.l2_get_relative_rates()


#%% Main
if __name__ == "__main__":
    file_path = 'data\\predictions\\german_credit_log_reg.csv'
    data = pd.read_csv(file_path)
    #data.head()

    fair = FairKit(
        y = data.credit_score, 
        y_hat = data.log_reg_pred, 
        a = data.sex, 
        r = data.log_reg_prob,
        model_type='Logistic Regression')
    #fair.create_fake_example()
    #fair.l2_plot()
    #fair.l2_ratio_subplot()

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

    file_path = 'data\\processed\\anonymous_data.csv'
    df = pd.read_csv(file_path)
    df.head()

    fair = FairKit(
        y = df.y, 
        y_hat = df.yhat, 
        a = df.grp, 
        r = df.phat,
        model_type='')

    fair.l2_plot()


