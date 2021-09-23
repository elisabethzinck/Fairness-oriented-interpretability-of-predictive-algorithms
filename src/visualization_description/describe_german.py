#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from src.visualization_description.descriptive_functions import DescribeData

figure_path = 'figures/descriptive_plots/'
#%% Load data
file_path_full = 'data\\processed\\german_credit_full.csv'
file_path = 'data\\processed\\german_credit.csv'

data = pd.read_csv(file_path)
data_full = pd.read_csv(file_path_full)

#%% See where single males lie in data
tmp = (data_full
    .assign(personal_status_sex = lambda x: x.sex + '_' + x.personal_status)
    .drop(columns = ['sex', 'personal_status']))
desc = DescribeData(y_name='credit_score', 
                    a_name = 'personal_status_sex',
                    id_name = 'person_id',
                    data = tmp)

desc.plot_tSNE(n_tries = 10)
plt.savefig(figure_path+'german_tsne_sex_status.pdf', bbox_inches='tight')

# %% Difference between sexes
desc = DescribeData(y_name='credit_score', 
                    a_name = 'sex',
                    id_name = 'person_id',
                    data = data)

desc.plot_tSNE(n_tries = 10)
plt.savefig(figure_path+'german_tsne_sex.pdf', bbox_inches='tight')
# %%
