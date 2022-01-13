# Create anonymous data used for presenting fairness toolkit

#%% Initialization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
input_path = 'data/predictions/german_credit_nn_pred.csv'
output_path = 'data/processed/anonymous_data.csv'

# %% Load data
german_data = pd.read_csv(input_path)
german_data.head()

# %% Creating new sensitive groups based on age
bins = [min(german_data.age), 25, 50, max(german_data.age)+1]
german_data['age_cat'] = pd.cut(
    german_data.age, 
    bins = bins, 
    include_lowest=True,
    right=False,
    labels=['A', 'B', 'C'])
german_data.age_cat.value_counts(dropna = False)

plt.hist(german_data.age, bins = range(15,80,5))
plt.axvline(x = bins[1], color = 'black')
plt.axvline(x = bins[2], color = 'black')
plt.xlabel('age')



#%% Renaming data and saving
name_map = {
    'age_cat': 'grp',
    'credit_score': 'y',
    'nn_pred': 'yhat',
    'nn_prob': 'phat',
}
anym = german_data[name_map.keys()].rename(columns = name_map)


anym.to_csv(output_path, index = None)
