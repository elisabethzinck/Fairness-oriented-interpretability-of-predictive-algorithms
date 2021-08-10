#%%
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

#%%

file_path = 'data\\processed\\german_credit.csv'
data = pd.read_csv(file_path)
data.head()
n = data.shape[0]

# Generate random classifications
preds = random.choices([0,1], k = n)
data['prediction'] = preds

# Do this in preprocessing instead? 
data['credit_score'] = data.credit_score - 1 

#%%
class EvaluationTool:
    def __init__(self, y, c, a):
        self.y = y
        self.c = c
        self.a = a

        self.classifier = pd.DataFrame({'y': y, 'a': a, 'c': c})
        self.sens_grps = self.a.unique()

    def get_confusion_matrix(self):
        counts = self.classifier \
            .groupby(['a', 'y', 'c']) \
            .size() \
            .reset_index(name = 'n')

        
        self.confusion = {}
        for grp in self.sens_grps:
            counts_grp = counts[counts.a == grp]
            TP = counts_grp.loc[(counts_grp.y == 1) & (counts_grp.c == 1), 'n']
            FP = counts_grp.loc[(counts_grp.y == 0) & (counts_grp.c == 1), 'n']
            TN = counts_grp.loc[(counts_grp.y == 0) & (counts_grp.c == 0), 'n']
            FN = counts_grp.loc[(counts_grp.y == 1) & (counts_grp.c == 0), 'n']
            self.confusion[grp] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
            
            # Convert counts to integer instead of pd.Series of length 1
            self.confusion[grp] = {
                key: int(count) 
                for key, count 
                in self.confusion[grp].items()}

        # Alternative way to do it (and much smarter...)
        self.cm = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            self.cm[grp] = confusion_matrix(
                y_true = df_group.y, 
                y_pred = df_group.c)
            
    def plot_confusion_matrix(self):
        if self.cm == None:
            self.get_confusion_matrix()
        
        n_grps = len(self.sens_grps)
        plt.figure(figsize = (15,5))
        for i, grp in enumerate(self.sens_grps):
            n_obs = sum(sum(self.cm[grp]))
            
            plt.subplot(1,2,i+1)
            sns.heatmap(
                self.cm[grp]/n_obs*100, 
                annot = True, 
                cmap = 'Blues', 
                vmin = 0, vmax = 100)
            plt.ylabel('Actual (%)')
            plt.xlabel('Predicted (%)')
            plt.title(f'{grp} ({n_obs} observations)')
        plt.show()
#%%

fair = EvaluationTool(
    y = data.credit_score, 
    c = data.prediction, 
    a = data.sex)
fair.get_confusion_matrix()
fair.plot_confusion_matrix()

#%%