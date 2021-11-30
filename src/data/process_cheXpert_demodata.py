#%%
import pandas as pd
import os
from src.models.data_modules import CheXpertDataModule

#%%
# Loading CheXpert
disease = 'Cardiomegaly'
dm = CheXpertDataModule(**{
    "target_disease": "Cardiomegaly", 
    'multi_label': False,
    "uncertainty_approach": "U-Zeros",
    'tiny_sample_data': False, 
    'extended_image_augmentation':False})

meta_dat = dm.dataset_df

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
df = (meta_dat
    .join(demo_df.set_index("patient_id"), how = "left", on = "patient_id")
    .assign(age_diff = lambda x: x.Age-x.age_at_CXR,
            sex_diff = lambda x: x.Sex != x.gender)
    .query("age_diff <= 15 and age_diff >= -15 and sex_diff == False")
    .filter(items = ["patient_id", "age_at_cxr", "gender", "race", "ethnicity"])
    .drop_duplicates()
)

#%%
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

df = df.assign(race_and_sex = [f"{df.gender.iloc[i]}_{df.race.iloc[i]}" for i in range(df.shape[0])])
# %% Writing processed file to csv 
save_dir = "data/CheXpert/processed/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)   

df.to_csv(save_dir+"cheXpert_processed_demo_data.csv", sep = ',', index = False)
