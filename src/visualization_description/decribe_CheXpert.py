#%%
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns

from src.evaluation_tool.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

fig_path_report = '../Thesis-report/00_figures/cheXpert/'
save_figs = True

#%%
# Choose dataset: test, train or val
data_set = "train"

# Loading CheXpert
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
        y = dm.val_data.y.squeeze()
    )
if data_set == "test":
    meta_dat = dm.test_data.dataset_df.assign(
        y = dm.test_data.y.squeeze()
)

#%%
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

df = df.assign(race_and_sex = [f"{df.Sex.iloc[i]}, {df.race.iloc[i]}" for i in range(df.shape[0])])

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
if save_figs: 
    plt.savefig(fig_path_report+f"posperc_sex_{data_set}.pdf", bbox_inches='tight')
desc_sex.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
if save_figs: 
    plt.savefig(fig_path_report+f"N_sex_{data_set}.pdf", bbox_inches='tight')
#%%
desc_race = DescribeData(a_name = "race", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

desc_race.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc_race.plot_positive_rate(title = f'Percentage with {disease}', orientation='v')
if save_figs: 
    plt.savefig(fig_path_report+f"posperc_race_{data_set}.pdf", bbox_inches='tight')
desc_race.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
if save_figs: 
    plt.savefig(fig_path_report+f"N_race_{data_set}.pdf", bbox_inches='tight')

#%%
desc_race_sex = DescribeData(a_name = "race_and_sex", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

desc_race_sex.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
desc_race_sex.plot_positive_rate(title = f'Percentage with {disease}', orientation='h')
if save_figs: 
    plt.savefig(fig_path_report+f"posperc_race_sex_{data_set}.pdf", bbox_inches='tight')
desc_race_sex.plot_n_target_across_sens_var(
    orientation='h',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
if save_figs: 
    plt.savefig(fig_path_report+f"N_race_sex_{data_set}.pdf", bbox_inches='tight')


# %%
