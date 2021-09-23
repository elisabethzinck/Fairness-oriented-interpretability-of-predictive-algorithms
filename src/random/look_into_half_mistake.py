#%%
import pandas as pd
import matplotlib.pyplot as plt
import torch
#%%
preds = pd.read_csv('data/predictions/german_credit_nn_pred.csv')
orig_data = pd.read_csv('data/processed/german_credit_full.csv')
#%%
plt.hist(preds.nn_prob, bins = 20)
preds = preds.assign(is_half = preds.nn_prob == 0.5)

#%% Is it the folds?
model_list = [torch.load(f'models/german_credit/NN_german_fold_{i}') for i in range(5)]


test_idx_df = pd.concat([
    pd.DataFrame({
        'test_idx': x['test_idx'], 
        'fold': x['fold']}) 
    for x in model_list])

check = pd.merge(preds, test_idx_df, left_on = 'person_id', right_on = 'test_idx')
check.groupby(['is_half', 'fold']).size().reset_index()

#%% It is sex?
preds.groupby(['is_half', 'sex']).size().reset_index()

#%% Is it status
alldata = pd.merge(preds, orig_data)
alldata = pd.merge(alldata, test_idx_df,left_on = 'person_id', right_on = 'test_idx')

alldata.groupby(['is_half', 'personal_status']).size().reset_index(name = 'n')

alldata.groupby('is_half').nunique()

problematic_data = alldata[alldata.is_half]
1+1
# %%
