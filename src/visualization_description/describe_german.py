#%%
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.evaluation_tool.descriptive_tool import DescribeData
from src.data.general_preprocess_functions import one_hot_encode_mixed_data

figure_path = 'figures/descriptive_plots/'
model_path = 'models/german_credit/'
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

# %% Make t-SNE plots of last layers in models
model_list = [torch.load(f'{model_path}/NN_german_fold_{i}') for i in range(5)]

# Get hidden neurons
i = 0

X = data_full.drop(['credit_score', 'person_id'], axis = 1)
X = one_hot_encode_mixed_data(X)
mod = model_list[i]['model']
test_idx = model_list[i]['test_idx']
train_idx = model_list[i]['train_idx']
X_train = data_full.iloc[train_idx]
X_test = data_full.iloc[test_idx]

scaler = StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)




#%%



#%%




