#%%
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
class EvaluationTool:
    def __init__(self, y, c, a, model_type = None, tol = 0.01):
        self.y = y
        self.c = c
        self.a = a
        self.model_type = model_type
        self.tol = tol

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'c': c})
        self.sens_grps = self.a.unique()

    def get_confusion_matrix(self):
        # Alternative way to do it (and much smarter...)
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
        if self.cm == None:
            self.get_confusion_matrix()
        
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
        if self.cm == None:
           self.get_confusion_matrix()
        self.rates = {}   
        for grp in self.sens_grps:
            TP, FN, FP, TN = self.cm[grp].values()
            self.rates[grp] = {'TPR': TP/(TP + FN), 
                               'FNR': FN/(TP + FN), 
                               'TNR': TN/(TN + FP), 
                               'FPR': FP/(TN + FP),
                               'PPV': TP/(TP + FP),
                               'FDR': FP/(TP + FP),
                               'NPV': TN/(TN + FN),
                               'FOR': FN/(TN + FN)
                               }
        return self.rates

    def false_negative_error_rate_balance(self):
        if self.rates == None:
           self.get_rates()
        FNR_list = [self.rates[grp]['FNR'] for grp in self.sens_grps]
        FNR_status = abs(np.diff(FNR_list).squeeze()) < self.tol
        return FNR_status 

    # aka equal opportunity
    def false_positive_error_rate_balance(self):
        if self.rates == None:
           self.get_rates()
        FPR_list = [self.rates[grp]['FPR'] for grp in self.sens_grps]
        FPR_status = abs(np.diff(FPR_list).squeeze()) < self.tol
        return FPR_status 

    # aka equalized odds 
    def separation(self):
        if self.rates == None:
           self.get_rates()
        if self.false_positive_error_rate_balance() and self.false_negative_error_rate_balance():
            return True 
        else:
            return False 



#%%
if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit.csv'
    data = pd.read_csv(file_path)
    data.head()
    n = data.shape[0]

    # Generate random classifications
    preds = random.choices([0,1], k = n)
    data['prediction'] = preds

    # Making logistic regression 
    log_reg = LogisticRegression(penalty='none')

    fair = EvaluationTool(
        y = data.credit_score, 
        c = data.prediction, 
        a = data.sex)
    fair.get_confusion_matrix()
    fair.plot_confusion_matrix()


#%%