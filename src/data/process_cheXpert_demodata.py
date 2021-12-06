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

meta_dat = dm.dataset_df.rename(columns = str.lower)

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
    .assign(age_diff = lambda x: x.age-x.age_at_CXR)
    .query("age_diff <= 15 and age_diff >= -15")
    .filter(items = ["patient_id", "age_at_cxr", "sex", "race", "ethnicity"])
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

mask = ~df.race.isin(['Asian', 'Black', 'White'])
df.loc[mask, "race"] = "Other"

df['race_sex'] = df.race + '_' + df.sex

# %% Writing processed file to csv 
save_dir = "data/CheXpert/processed/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)   

df.to_csv(save_dir+"cheXpert_processed_demo_data.csv", sep = ',', index = False)
