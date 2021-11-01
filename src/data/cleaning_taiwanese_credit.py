#%%
import pandas as pd
import matplotlib.pyplot as plt

from src.data.general_preprocess_functions import map_several_columns

#%%
input_path = 'data/raw/taiwanese_credit/default of credit card clients.xls'
output_path = 'data/processed/taiwanese_credit.csv'

taiwan_credit_raw = pd.read_excel(input_path, skiprows=1)
data = taiwan_credit_raw.copy() # To save downloading time


# %% Renaming columns
data.columns = [colname.lower() for colname in data.columns]
data.rename(
    columns = {'default payment next month': 'default_next_month'},
    inplace = True)

#%% Map categorical variables to strings
# Variables originally encoded using numbers
# education and marriage map don't correspond to explanation in article, 
# instead map from  https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/discussion/34608 is used
sex_map = {
    1: 'Male',
    2: 'Female'
}
education_map = {
    0: 'others',
    1: 'graduate school',
    2: 'university',
    3: 'high school',
    4: 'others',
    5: 'others',
    6: 'others'
    } 
marriage_map = {
    0: 'others',
    1: 'married',
    2: 'single',
    3: 'others'} 

map_dict = {'sex': sex_map, 'education': education_map, 'marriage': marriage_map}
data = map_several_columns(data, map_dict)

# %% Final checks before writing
assert data.shape == (30000, 25)
assert data.id.nunique() == 30000
assert data.isnull().sum().sum() == 0

# %% Write data
data.to_csv(output_path, index = False)

