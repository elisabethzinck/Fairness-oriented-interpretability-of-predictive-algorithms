#%%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import myData

#%%
# Inputs
dataset = 1 # 1 or 2 (which field strength of MRI)
time_horizon = '5y'

id_var = 'rid'
sens_vars = ['sex', 'age']
y_var = 'label'


# Load data
raw_data = {}
patient_groups = ['ad', 'mci', 'nc']
for pat_grp in patient_groups:
    file_path = f'data/ADNI/raw/sorted_{pat_grp}{dataset}.csv'
    raw_data[pat_grp] = pd.read_csv(file_path)
    raw_data[pat_grp].columns = raw_data[pat_grp].columns.str.lower()

# Setup
# Create all data
ad = raw_data['ad'].assign()


# %%
