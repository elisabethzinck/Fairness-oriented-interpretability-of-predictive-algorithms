# Descriptive stuff 

#%%
from numpy.core.defchararray import count
from numpy.core.numeric import count_nonzero
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from seaborn.axisgrid import FacetGrid 

from src.evaluation_tool.layered_tool import FairKit

#%%

if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit.csv'
    pred_filte_path = 'data\\predictions\\german_credit_log_reg.csv'

    data = pd.read_csv(file_path)
    pred_data = pd.read_csv(pred_filte_path)

    fair = FairKit(
        y = pred_data.credit_score, 
        y_hat = pred_data.log_reg_pred, 
        a = pred_data.sex, 
        r = pred_data.log_reg_prob,
        model_type='Logistic Regression')

    fair.l1_get_data().drop(columns = ['avg_w_error'])

    grp_data = (data.groupby(by=['sex', 'credit_score'])
                    .agg(n=('person_id', 'count'))
                    .groupby(by='sex')
                    .apply(lambda x: x/x.sum())
                    .reset_index()
                )
    grp_data.rename({'n':'fraction'}, axis=1, inplace=True)

    fig = plt.figure(figsize = (8,5))
    gs = GridSpec(nrows = 1, ncols = 1)
    ax = fig.add_subplot(gs[0,0])
    sns.barplot(x='credit_score',
                y='fraction',
                hue = 'sex',
                data = grp_data, 
                ax = ax)
    ax.set_ylim((0,1))
    plt.show()

# %%
