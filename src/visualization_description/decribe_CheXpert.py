#%%
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns

from src.evaluation_tool.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

# Distributions without considering patients with more than one image

#%% Aggregated tables for Cardiomelagy  

# Choose dataset: test, train or val
data_set = "test"




disease = 'Cardiomegaly'
dm = CheXpertDataModule(**{
    "target_disease": "Cardiomegaly", 
    'multi_label': False,
    "uncertainty_approach": "U-Zeros",
    'tiny_sample_data': False, 
    'extended_image_augmentation':False})

if data_set == "train":
    meta_dat = dm.train_data.dataset_df.assign(
        y = dm.train_data.y.squeeze()
    )
if data_set == "val":
    meta_dat = dm.val_data.dataset_df.assign(
        y = dm.train_data.y.squeeze()
    )
if data_set == "test":
    meta_dat = dm.val_data.dataset_df.assign(
        y = dm.train_data.y.squeeze()
)

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

# Adding Race 
raw_demo = pd.read_excel('data/CheXpert/raw/chexpertdemodata-2/CHEXPERT DEMO.xlsx')
demo_df = raw_demo.rename(columns={
    "PATIENT":"patient_id", 
    "PRIMARY_RACE": "race",
    "ETHNICITY": "ethnicity",
    "GENDER": "gender", 
    "AGE_AT_CXR": "age_at_CXR"
    })
df = meta_dat.join(demo_df.set_index("patient_id"), how = "left", on = "patient_id")

# Processing race as they have done in CheXploration
# https://github.com/biomedia-mira/chexploration/blob/main/notebooks/chexpert.sample.ipynb 
mask = (df.race.str.contains("Black", na=False))
df.loc[mask, "race"] = "Black"

mask = (df.race.str.contains("White", na=False))
df.loc[mask, "race"] = "White"

mask = (df.race.str.contains("Asian", na=False))
df.loc[mask, "race"] = "Asian"

# Filtering to only include Black, White and Asian
df = df[df.ethnicity.isin(["Non-Hispanic/Non-Latino","Not Hispanic"])]
df = df[df.race.isin(["Black", "White", "Asian"])]

df = df.assign(race_and_sex = [f"{df.Sex.iloc[i]}_{df.race.iloc[i]}" for i in range(df.shape[0])])

#%%
# Descriptive plots: 
desc_sex = DescribeData(a_name = "Sex", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

desc_sex.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc_sex.plot_positive_rate(title = f'Percentage with {disease}', orientation='v')
desc_sex.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
#%%
desc_race = DescribeData(a_name = "race", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

desc_race.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc_race.plot_positive_rate(title = f'Percentage with {disease}', orientation='v')
desc_race.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})

#%%
desc_race_sex = DescribeData(a_name = "race_and_sex", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

desc_race_sex.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc_race_sex.plot_positive_rate(title = f'Percentage with {disease}', orientation='h')
desc_race_sex.plot_n_target_across_sens_var(
    orientation='h',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})

#%%
count_race = pd.DataFrame(df.race.value_counts()).reset_index().rename(columns={"race": "Count", "index": "Race"})
plt.figure(figsize=(10,10))
sns.barplot(x="Count", y = "Race", data = count_race)

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
pd.read_csv("data/CheXpert/predictions/test_model/predictions.csv")