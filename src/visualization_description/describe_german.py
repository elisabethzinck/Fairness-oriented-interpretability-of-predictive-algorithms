#%%
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.visualization_description.descriptive_tool import DescribeData
from src.data.general_preprocess_functions import one_hot_encode_mixed_data

figure_path = 'figures/descriptive_plots/'
fig_path_report = '../Thesis-report/00_figures/describe_data/'
model_path = 'models/german_credit/'

update_report_figs = False

run_t_sne = False

#%% Load data
file_path = 'data\\processed\\german_credit.csv'

data = pd.read_csv(file_path)

#%% Plot number of bad payers by sensitive variable 
desc = DescribeData(y_name='credit_score', 
                    a_name = 'sex',
                    id_name = 'person_id',
                    data = data, 
                    data_name = 'German Credit Score')

desc.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=True, 
    **{"class_0_label":"Good", "class_1_label":"Bad", "legend_title":"Credit Score"})
if update_report_figs: 
    plt.savefig(fig_path_report+'german_N_by_sex.pdf', bbox_inches='tight')

desc.descriptive_table_to_tex(target_tex_name='Bad Credit Score')

#%% See where single males lie in data
tmp = (data_full
    .assign(personal_status_sex = lambda x: x.sex + '_' + x.personal_status)
    .drop(columns = ['sex', 'personal_status']))
desc = DescribeData(y_name='credit_score', 
                    a_name = 'personal_status_sex',
                    id_name = 'person_id',
                    data = tmp)
if run_t_sne:
    desc.plot_tSNE(n_tries = 10)
    plt.savefig(figure_path+'german_tsne_sex_status.pdf', bbox_inches='tight')

# %% Difference between sexes
desc = DescribeData(y_name='credit_score', 
                    a_name = 'sex',
                    id_name = 'person_id',
                    data = data)
if run_t_sne:
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




