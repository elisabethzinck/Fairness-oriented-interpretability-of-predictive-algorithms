#%%
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt 

from src.evaluation_tool.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

# Distributions without considering patients with more than one image

#%% Aggregated tables for Cardiomelagy  
disease = 'Cardiomegaly'
dm = CheXpertDataModule(**{
    "target_disease": disease})
meta_dat = dm.train_data.dataset_df

desc = DescribeData(a_name = "Sex", 
                    y_name = "y", 
                    id_name = 'patient_id', 
                    data = meta_dat,
                    data_name=f'CheXpert, target: {disease}', 
                    **{"decimal":4})

desc.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc.plot_positive_rate(title = f'Percentage with {disease}', orientation='v')
desc.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
# %% CheXpert race data 
dm = CheXpertDataModule(**{
    "target_disease": "Cardiomegaly", 
    'multi_label': False,
    "uncertainty_approach": "U-Zeros",
    'tiny_sample_data': False, 
    'extended_image_augmentation':False})
all_df = dm.dataset_df.filter(items = ["patient_id", "Sex", "Age"])
raw_demo = pd.read_excel('data/CheXpert/raw/chexpertdemodata-2/CHEXPERT DEMO.xlsx')
demo_df = raw_demo.rename(columns={"PATIENT":"patient_id"})
# %%
df = all_df.join(demo_df.set_index("patient_id"), how = "left", on = "patient_id")

# Checking if age is the same in data sets 
df = df.assign(age_diff = lambda x: x.Age - x.AGE_AT_CXR,
gender_diff = lambda x: x.Sex != x.GENDER)

# %%
vc = df.patient_id.value_counts()
recur_pat = vc[vc > 1].index
recur_df = df.loc[df.patient_id.isin(recur_pat)]

#%%
l = []
for i, patient in enumerate(df.patient_id.unique()):
    age_exact = (df.query(f"patient_id == '{patient}'").age_diff == 0).sum()
    if age_exact < 1: 
        l.append(patient)
# %%

age_diff_df = df.loc[df.patient_id.isin(l)] 
# 8135 patients with age diff between datasets


#%% Checking gender diff
df.query("gender_diff == True").dropna()
# Three people left who have a difference in sex/gender 
# %%
plot_df = (age_diff_df.query('age_diff <= 15 and age_diff >= -15')
    .assign(age_diff = lambda x: [int(x.age_diff.iloc[i]) for i in range(x.shape[0])])

)
plt.hist(x = 'age_diff', data = plot_df, bins=31)
# %%
