#%%
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, roc_curve

from src.evaluation_tool.utils import (
    cm_matrix_to_dict, cm_dict_to_matrix, abs_percentage_tick)

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

        self.get_confusion_matrix()
        self.get_rates()

        # Define color palette
        n_grps = len(self.sens_grps)
        cols = sns.color_palette(n_colors = n_grps)
        self.sens_grps_cols = dict(zip(self.sens_grps, cols))

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
        df = (pd.DataFrame(self.cm)
            .T.reset_index()
            .rename(columns = {'index':'grp'})
            .assign(
                n = lambda x: x.TP + x.FN + x.FP + x.TN,
                PP = lambda x: x.TP + x.FP,
                avg_w_error = lambda x: (w_fp*x.FP + (1-w_fp)*x.FN)/x.n))
        min_err = min(df.avg_w_error)
        df['perc_diff'] = (df.avg_w_error-min_err)/abs(min_err)*100

        return df

    def l1_plot(self):
        pass

    def l1_calculate(self):
        pass

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
        axis.tick_params(left=False)
        axis.legend(loc = 'upper right', frameon = False)

        return axis

    def l2_ratio_subplot(self, axis = None):
        discrim_rates = ['FPR', 'FNR', 'FDR', 'FOR']
        name_map = {}
        rel_rates = pd.DataFrame({'rate': discrim_rates})
        for i, grp in enumerate(self.sens_grps):
            grp_lab = 'grp' + str(i)
            name_map[grp_lab] = grp
            rel_rates[grp_lab] = [
                self.rates[grp][rate] for rate in rel_rates.rate
                ]

        rel_rates = rel_rates.assign(
            grp0_ratio = lambda x: (x.grp0 - x.grp1)/abs(x.grp1)*100,
            grp1_ratio = lambda x: (x.grp1 - x.grp0)/abs(x.grp0)*100,
            abs_ratio = lambda x: np.maximum(x.grp0_ratio, x.grp1_ratio)
        )

        rel_rates['max_grp'] = np.argmax(
            np.array(rel_rates[['grp0_ratio', 'grp1_ratio']]),
            axis=1)
        rel_rates['signed_ratio'] = rel_rates.abs_ratio*(rel_rates.max_grp*2-1)
        rel_rates['max_grp'] = [
            name_map['grp'+str(grp)] for grp in rel_rates.max_grp]

        if axis is None:
            axis = plt.gca()
        sns.barplot(
            x = 'signed_ratio', y = 'rate', 
            hue = 'max_grp', palette = self.sens_grps_cols,
            data = rel_rates,
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


    def l2_plot(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        self.l2_ratio_subplot(axis = ax1)
        self.l2_rate_subplot(axis = ax2)
        f.subplots_adjust(wspace = 0.5, right = 1)

    def create_fake_example(self):
        """Modifies rate to show discrimination of women"""
        self.rates['female']['FPR'] = 0.47




#%% Main
if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit_pred.csv'
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
    fair.l2_ratio_subplot()

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