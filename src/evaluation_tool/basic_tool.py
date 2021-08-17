#%%
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pprint
import plotnine as p9


#%% Helper functions
def max_abs_diff(l):
    """Returns the maximum pairwise absolute difference between the elements in l"""
    out = max([abs(x-y) for x in l for y in l])
    return(out)

#%%
class EvaluationTool:
    def __init__(self, y, c, a, model_type = None, tol = 0.03):
        self.y = y
        self.c = c
        self.a = a
        self.model_type = model_type
        self.tol = tol

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'c': c})
        self.sens_grps = self.a.unique()

        self.get_confusion_matrix()
        self.get_rates()

    def get_confusion_matrix(self):
        self.cm_sklearn = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            self.cm_sklearn[grp] = confusion_matrix(
                y_true = df_group.y, 
                y_pred = df_group.c)
        
        self.cm = {}
        for grp in self.cm_sklearn.keys():
            TN, FP, FN, TP = self.cm_sklearn[grp].ravel()
            self.cm[grp] = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
        
        return self.cm
            
    def plot_confusion_matrix(self):
        plt.figure(figsize = (15,5))
        if self.model_type != None:
            plt.suptitle(f'Model: {self.model_type}')
        for i, grp in enumerate(self.sens_grps):
            n_obs = sum(self.cm[grp].values())
            grp_cm = np.array(list(self.cm[grp].values())).reshape(2,2)
            
            plt.subplot(1,2,i+1)
            ax = sns.heatmap(
                grp_cm/n_obs*100, 
                annot = True, 
                cmap = 'Blues', 
                vmin = 0, vmax = 100,
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'], 
                annot_kws={'size':15})
            for a in ax.texts: a.set_text(f"{a.get_text()}%")
            plt.ylabel('Actual (%)')
            plt.xlabel('Predicted (%)')
            plt.title(f'{str.capitalize(grp)} ({n_obs} observations)')
        plt.show()

    def get_rates(self):
        self.rates = {}   
        for grp in self.sens_grps:
            TP, FN, FP, TN = self.cm[grp].values()
            self.rates[grp] = {'PR': (TP + FP)/(TP + FN + FP + TN),
                               'TPR': TP/(TP + FN), 
                               'FNR': FN/(TP + FN), 
                               'TNR': TN/(TN + FP), 
                               'FPR': FP/(TN + FP),
                               'PPV': TP/(TP + FP),
                               'FDR': FP/(TP + FP),
                               'NPV': TN/(TN + FN),
                               'FOR': FN/(TN + FN)
                               }
        return self.rates

    def get_rates_overview(self):
        rates_overview = pd.DataFrame(
            [["PR", "1-NR"],
            ["TPR", "1-FNR"],
            ["TNR", "1-FPR"],
            ["PPV", "1-FDR"],
            ["NPV", "1-FOR"]],
            columns = ["rate", "rate_equiv"]
        )

        for grp in fair.sens_grps:
            rates_overview[grp] = [
                fair.rates[grp][rate] 
                for rate in rates_overview.rate]
        
        # To do: Code diff for != 2 sens_grps
        rates_overview['abs_diff'] = abs(rates_overview[fair.sens_grps[0]] - rates_overview[fair.sens_grps[1]])
        self.rates_overview = rates_overview
        return rates_overview


    def get_obs_crit(self):
        fair.obs_crit = {}
        all_obs_crit = {
            'independence': ['PR'],
            'separation': ['FPR', 'FNR'],
            'false_positive_error_rate': ['FPR'],
            'false_negative_error_rate': ['FNR'],
            'sufficiency': ['PPV', 'NPV'],
            'predictive_parity': ['NPV']}


        for crit in all_obs_crit:
            fair.obs_crit[crit] = {}
            fair.obs_crit[crit]['values'] = {}
            fair.obs_crit[crit]['max_diff'] = 0
            for val in all_obs_crit[crit]:
                fair.obs_crit[crit]['values'][val] = {}
                for grp in fair.sens_grps:
                    fair.obs_crit[crit]['values'][val][grp] = fair.rates[grp][val]
                
                diff = max_abs_diff(fair.obs_crit[crit]['values'][val].values())
                fair.obs_crit[crit]['max_diff'] = max(diff, fair.obs_crit[crit]['max_diff'])
            fair.obs_crit[crit]['passed'] = fair.obs_crit[crit]['max_diff'] < fair.tol
        
        return fair.obs_crit

    def plot_rates(self):
        tol_ribbon = pd.DataFrame({
            'x':[0,1], 
            'ymin':[0-self.tol, 1-self.tol],
            'ymax':[0+self.tol, 1+self.tol]})
        grp0 = self.sens_grps[0]
        grp1 = self.sens_grps[1]
        xy_min = min(self.rates_overview.min()[[grp0,grp1]])
        xy_max = max(self.rates_overview.max()[[grp0,grp1]])
        xy_lims = (xy_min-0.1, xy_max+0.1)

        p = p9.ggplot() + \
            p9.geom_point(
                self.rates_overview, p9.aes(x=grp0, y=grp1), 
                color = 'steelblue', size = 4) + \
            p9.labs(
                title = f'Rates: {str.capitalize(grp0)} vs. {str.capitalize(grp1)}',
                x = str.capitalize(grp0), 
                y = str.capitalize(grp1)) + \
            p9.geom_ribbon(
                tol_ribbon, p9.aes(x='x', ymin='ymin', ymax='ymax'), 
                alpha = 0.2) + \
            p9.geom_abline(
                p9.aes(intercept = 0, slope = 1), 
                color = 'grey', linetype = 'dashed') + \
            p9.coord_cartesian(xlim = xy_lims, ylim = xy_lims) + \
            p9.theme_minimal() + \
            p9.scale_color_brewer(type='qual', palette=8, direction=1) + \
            p9.geom_text(fair.rates_overview,
                        p9.aes(x=grp0, y=grp1, label='rate'),
                        color='black',
                        size=9, 
                        alpha=0.8, 
                        nudge_y=.02) +\
            p9.theme(legend_position='none',
                    figure_size=(5,5), 
                    aspect_ratio=1) 
        return p



#%%
if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit_pred.csv'
    data = pd.read_csv(file_path)
    data.head()

    fair = EvaluationTool(
        y = data.credit_score, 
        c = data.logistic_regression_prediction, 
        a = data.sex, 
        model_type='Logistic Regression')

    fair.plot_confusion_matrix()
    print(fair.get_rates_overview())
    obs_crit = fair.get_obs_crit()

    pprint.pprint(obs_crit)
    p = fair.plot_rates()
    p
      

# %%

