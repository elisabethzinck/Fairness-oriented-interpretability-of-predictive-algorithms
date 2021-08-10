
import pandas as pd
import random

file_path = 'data\\processed\\german_credit.csv'
data = pd.read_csv(file_path)
data.head()
n = data.shape[0]

# Generate random classifications
preds = random.choices([0,1], k = n)
data['prediction'] = preds

# 
# Do this in preprocessing instead? 
data['credit_score'] = data.credit_score - 1 


class EvaluationTool:
    def __init__(self, y, c, a):
        self.y = y
        self.c = c
        self.a = a
        self.classifier = pd.DataFrame({'y': y, 'a': a, 'c': c})

    def get_confusion_matrix(self):
        self.sens_grps = self.a.unique()

        counts = self.classifier \
            .groupby(['a', 'y', 'c']) \
            .size() \
            .reset_index(name = 'n')

        
        self.confusion = {}
        for grp in sens_grps:
            counts_gpr = counts[counts.a == grp]
            TP = counts_grp.loc[(counts_grp.y == 1) & (counts_grp.c == 1), 'n']
            FP = counts_grp.loc[(counts_grp.y == 0) & (counts_grp.c == 1), 'n']
            TN = counts_grp.loc[(counts_grp.y == 0) & (counts_grp.c == 0), 'n']
            FN = counts_grp.loc[(counts_grp.y == 1) & (counts_grp.c == 0), 'n']
            self.confusion[grp] = {'TP': TP, 'FP':FP, 'TN': TN, 'FN': FN}




    def evaluate_independence(self):
        pass

    def evaluate_sufficiency(self):
        pass

    def evaluate_separation(self):
        pass

german_eval = EvaluationTool(
    y = data.credit_score, 
    c = data.prediction, 
    a = data.sex)

german_eval.get_confusion_matrix()
1+1

counts = german_eval.classifier \
    .groupby(['a', 'y', 'c']) \
    .size() \
    .reset_index(name = 'n')

grp = 'male'
counts.dtypes
counts_grp = counts[counts.a == grp]
counts_grp
# To do: make this a "normal" int instead of a series
TP = counts_grp.loc[(counts_grp.y == 1) & (counts_grp.c == 1),'n']
type(TP)