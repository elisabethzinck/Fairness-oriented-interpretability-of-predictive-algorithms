#%%
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        
        # extracting sklearn metrics to dict 
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
        return rates_overview


    def get_obs_crit(self):
        fair.obs_crit = {}
        all_obs_crit = {
            'independence': ['PR'],
            # To do: Include conditional statistical parity? 
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



#%%
if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit_pred.csv'
    data = pd.read_csv(file_path)
    data.head()

    fair = EvaluationTool(
        y = data.credit_score, 
        c = data.logistic_regression_prediction, 
        a = data.sex)

    fair.plot_confusion_matrix()
    print(fair.get_rates_overview())
    obs_crit = fair.get_obs_crit()

    pprint.pprint(obs_crit)
# %%
