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
# %%
