#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from src.visualization_description.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

##################################################
# Investigating datamodule vs. demo data 
###################################################
# Loading CheXpert
disease = 'Cardiomegaly'
dm = CheXpertDataModule(**{
    "target_disease": "Cardiomegaly", 
    'multi_label': False,
    "uncertainty_approach": "U-Zeros",
    'tiny_sample_data': False, 
    'extended_image_augmentation':False})

# %%
meta_dat = dm.dataset_df.rename(columns = str.lower)
target_map = {
    np.nan: 0,  # unmentioned
    0.0: 0,     # negative
    -1.0: 1,    # uncertain
    1.0: 1      # positive
    }
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
    .assign(
        y = lambda x: x.cardiomegaly.map(target_map),
        age_diff = lambda x: x.age-x.age_at_CXR)
    .query("age_diff <= 15 and age_diff >= -15")
    .filter(items = ["patient_id", "age_at_cxr", "sex", "race", "ethnicity", "y"])
)

#%%
# We have 200 unique patient ids in the val set
cheXpert_patients = set(meta_dat.patient_id.unique())
demo_dat_patients = set(demo_df.patient_id.unique())

patients_not_in_demo = cheXpert_patients.difference(demo_dat_patients)

p_not_in_demo_dat = meta_dat[meta_dat.patient_id.isin(list(patients_not_in_demo))]

(meta_dat.shape[0]-df.shape[0]-17) == p_not_in_demo_dat.shape[0]

# %%

##################################################
# Investigating raw CheXpert vs. demo data 
###################################################
cheXpert_raw_val = (pd.read_csv('data/CheXpert/raw/CheXpert-v1.0-small/valid.csv')
    .assign(
        patient_id = lambda x: x.Path.str.split('/').str[2],
        Path = lambda x: "data/CheXpert/raw/" + x.Path)
)
cheXpert_raw = (pd.read_csv('data/CheXpert/raw/CheXpert-v1.0-small/train.csv')
    .assign(
        patient_id = lambda x: x.Path.str.split('/').str[2],
        Path = lambda x: "data/CheXpert/raw/" + x.Path)
)

224316-cheXpert_raw.shape[0]
224316-meta_dat.shape[0]

cheXpert_raw_patients = set(cheXpert_raw.patient_id.unique())
print(f"Patients in raw: {len(cheXpert_raw_patients)}")
print(f"Patients in dm: {len(cheXpert_patients)}")
print(f"Diff #patients between raw and dm: {len(cheXpert_raw_patients)-len(cheXpert_patients)}")
print(f"Patients in demo: {len(demo_dat_patients)}")

# %% Patients by views 
front_df = cheXpert_raw[cheXpert_raw["Frontal/Lateral"] == 'Frontal']
lat_df = cheXpert_raw[cheXpert_raw["Frontal/Lateral"] == 'Lateral']

lat_patients = set(lat_df.patient_id.unique())
front_patients = set(front_df.patient_id.unique())

print(f"Patients with only lateral scans: {len(lat_patients.difference(front_patients))}")


# %%
