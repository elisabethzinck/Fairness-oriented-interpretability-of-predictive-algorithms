#%% Imports
import pandas as pd
import numpy as np
import math

# Widgets
from ipywidgets import interactive, fixed
import ipywidgets as widgets
from IPython.display import display

# Plots 
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go

# sklearn 
from sklearn.metrics import confusion_matrix, roc_curve

# dir functions
from src.evaluation_tool.utils import (
    cm_matrix_to_dict, cm_dict_to_matrix, abs_percentage_tick, flatten_list)

#%%
def get_minimum_rate(group):
    """Helper function to calculate minimum rate by group"""
    group['min_rate'] = group['rate_val'].agg('min')
    return group

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

        self.cm = self.get_confusion_matrix()
        self.rates = self.get_rates()
        self.rel_rates = self.get_relative_rates()

        # Define color palette
        cols = sns.color_palette(n_colors = self.n_sens_grps)
        self.sens_grps_cols = dict(zip(self.sens_grps, cols))

    def get_confusion_matrix(self):
        """Calculate the confusion matrix for sensitive groups"""
        if hasattr(self, 'cm'):
            return self.cm
        else:
            cm = {}
            for grp in self.sens_grps:
                df_group = self.classifier[self.classifier.a == grp]
                cm_sklearn = confusion_matrix(
                    y_true = df_group.y, 
                    y_pred = df_group.y_hat)
                cm[grp] = cm_matrix_to_dict(cm_sklearn)
            return cm



    def get_rates(self):
        """Calculate rates by senstitive group"""
        if hasattr(self, 'rates'):
            return self.rates
        else:
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
                    'PN/n': (TN+FN)/(TP+FP+TN+FN)
                    }
            return rates

    def get_relative_rates(self):
        """Calculate relative difference in rates between group rate 
        and minimum rate. Dataframe used for ratio plot in 
        second layer of evaluation
        """
        if hasattr(self, 'rel_rates'):
            return self.rel_rates
        else:
            discrim_rates = ['FPR', 'FNR', 'FDR', 'FOR', 'PN/n']
        
            # Convert rate dict to data frame
            rates_df = pd.DataFrame(
                [(grp, rate, val) 
                for grp,grp_dict in self.rates.items() 
                for rate,val in grp_dict.items()], 
                columns = ['grp', 'rate', 'rate_val'])
            
            # Calculate relative rates
            rel_rates = (rates_df[rates_df.rate.isin(discrim_rates)]
                .groupby(by = 'rate')
                .apply(get_minimum_rate)
                .assign(
                    rate_ratio = lambda x: 
                        (x.rate_val-x.min_rate)/x.min_rate*100))

            return rel_rates

    def l1_get_data(self, w_fp = 0.5):
        """Get data used for first layer of evaluation
        
        Args:
            w_fp (float): False positive error weight
        """
        df = (pd.DataFrame(self.cm)
            .T.reset_index()
            .rename(columns = {'index': 'group'})
            .assign(
                n = lambda x: x.TP + x.FN + x.FP + x.TN,
                percent_positive = lambda x: (x.TP + x.FN)/x.n*100,
                avg_w_error = lambda x: (w_fp*x.FP + (1-w_fp)*x.FN)/(2*x.n)))
        min_err = min(df.avg_w_error)
        df['unfair'] = (df.avg_w_error-min_err)/abs(min_err)*100

        # Make table pretty
        cols_to_keep = ['group', 'unfair', 'avg_w_error', 'n', 'percent_positive']
        digits = {'unfair': 1, 'avg_w_error': 3, 'percent_positive': 1}
        df = (df[cols_to_keep]
            .round(digits))

        return df


    def l2_rate_subplot(self, ax = None, w_fp = 50):
        """Plot FPR, FNR, FDR, FOR for each group"""
        rate_order = ['FPR', 'FNR', 'FDR', 'FOR']
        plot_df = self.rel_rates[self.rel_rates.rate != 'PN/n']

        if w_fp >= 0.5:
            # List in order [FPR, FNR, FDR, FOR] to match plot order
            weight_list = [1, 1.05-w_fp, 1, 1.05-w_fp] 
        else:
            # List in order [FPR, FNR, FDR, FOR] to match plot order
            weight_list = [.5+w_fp, 1, .5+w_fp, 1]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x = 'rate_val', y = 'rate', 
            hue = 'grp', palette = self.sens_grps_cols,
            data = plot_df,
            order = rate_order,
            ax = ax)
        ax.legend(loc = 'upper right', frameon = False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(0,1)
        for pos in ['right', 'top', 'left']:
            ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, labelsize=12)
        if w_fp != 0.5:
            containers = flatten_list([list(ax.containers[i][0:4]) for i in range(self.n_sens_grps)])
            for bar, alpha in zip(containers, weight_list*self.n_sens_grps):
                bar.set_alpha(alpha)

        return ax

    def l2_ratio_subplot(self, ax = None, w_fp = 0.5):
        """Plot the rate ratio for each sensitive groups
        
        Args:
            w_fp (float in (0,1)): False positive error weight
        
        """
        w_size = 1000
        rate_positions = {'FPR': 1, 'FNR': 0.75, 'FDR': 0.5, 'FOR': 0.25}
        rate_weights = {
            'FPR': w_size*w_fp, 'FNR': w_size*(1-w_fp), 
            'FDR': w_size*w_fp, 'FOR': w_size*(1-w_fp)}
        plot_df = (self.rel_rates
            .query("rate != 'PN/n'")
            .assign(
                rate_position = lambda x: x.rate.map(rate_positions), point_size = lambda x: x.rate.map(rate_weights)))
        size_tuple=(200*(1-w_fp), 200*w_fp)
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.hlines(y=plot_df.rate_position,
                xmin = 0,
                xmax = plot_df.rate_ratio,
                color = 'lightgrey', 
                alpha = 1, 
                linestyles='solid', 
                linewidth=1, 
                zorder = 1)
        sns.scatterplot(data = plot_df, 
                        x = 'rate_ratio',
                        y='rate_position',
                        hue='grp',
                        palette = self.sens_grps_cols, 
                        size = 'point_size',
                        sizes = size_tuple, #(100,200),
                        legend = False,
                        ax = ax,
                        marker = 'o',
                        alpha = 1, 
                        zorder = 2)
        ax.set_yticks(list(rate_positions.values()))
        ax.set_yticklabels(list(rate_positions.keys()))
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(
            'Relative Difference of Group Rate vs. Minimum Group Rate', fontsize=14) 
        ax.set_ylim((.125,1.125))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
        for pos in ['right', 'top', 'left']:
                    ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, labelsize=12)
    

    def l2_plot(self, w_fp = 0.5):
        gs = GridSpec(nrows = 3, ncols = 1)
        f = plt.figure(figsize=(8,6))
        ax_list = [f.add_subplot(gs[0:2,0]),
                    f.add_subplot(gs[2,0])]
        self.l2_rate_subplot(ax = ax_list[0], w_fp = w_fp)
        self.l2_ratio_subplot(ax = ax_list[1], w_fp = w_fp)
        ax_list[0].set_title('Group Rates', fontsize=14)
        ax_list[0].legend(title='Group', frameon = False)
        f.subplots_adjust(hspace = 0.8, right = 1)

    def l2_interactive_plot(self):
       return interactive(self.l2_plot, w_fp=(0,1,0.1))


    def l3_plot_fairness_criteria(self):
        fairness_crit = pd.DataFrame([
            ['independence', 'PN/n'],
            ['separation', 'FPR'],
            ['separation', 'FNR'],
            ['false_positive_error_rate_balance', 'FPR'],
            ['equal_opportunity', 'FNR'],
            ['sufficiency', 'FDR'],
            ['sufficiency', 'FOR'],
            ['predictive_parity', 'FDR']],
            columns = ['criterion', 'rate'])

        plot_df = (
            pd.merge(
                self.rel_rates[['rate', 'rate_ratio']], 
                fairness_crit)
            .groupby(by = ['rate', 'criterion'], as_index = False)
            .agg('max') # Get maximum of the rate ratio of sensitive groups
            .drop('rate', axis = 1)
            .groupby(by = 'criterion', as_index = False)
            .agg('mean')) # Get mean of rate ratio by rates

        criteria_order = (plot_df
            .sort_values('rate_ratio', ascending = False)
            .criterion
            .tolist())

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x = 'rate_ratio', y = 'criterion', 
            data = plot_df,
            order = criteria_order,
            color = 'grey',
            ax = ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        for pos in ['right', 'top', 'left']:
            ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, labelsize=12)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))

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
    fair.l2_plot(w_fp=0.7)
    fair.l3_plot_fairness_criteria()

    compas_file_path = 'data\\processed\\compas\\compas-scores-two-years-pred.csv'
    compas = pd.read_csv(compas_file_path)
    compas.head()

    fair_compas = FairKit(
        y = compas.two_year_recid, 
        y_hat = compas.pred_medium_high, 
        a = compas.age_cat, 
        r = compas.decile_score,
        model_type='COMPAS Decile Scores')
    fair_compas.l2_plot()

    file_path = 'data\\processed\\anonymous_data.csv'
    df = pd.read_csv(file_path)
    df.head()

    fair_anym = FairKit(
        y = df.y, 
        y_hat = df.yhat, 
        a = df.grp, 
        r = df.phat,
        model_type='')
    fair_anym.l2_plot(w_fp=0.7)
    #plt.savefig('../Thesis-report/00_figures/L2_example.pdf', bbox_inches='tight')

# %%





