#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from src.evaluation_tool.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

fig_path_report = '../Thesis-report/00_figures/cheXpert/'
save_figs = True

#%%
# Choose dataset: test, train or val
data_set = "all"

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
if data_set == "all":
     # Uncertainty approach
    if dm.uncertainty_approach == 'U-Ones':
        target_map = {
            np.nan: 0,  # unmentioned
            0.0: 0,     # negative
            -1.0: 1,    # uncertain
            1.0: 1      # positive
            }
    elif dm.uncertainty_approach == 'U-Zeros':
        target_map = {
            np.nan: 0,  # unmentioned
            0.0: 0,     # negative
            -1.0: 0,    # uncertain
            1.0: 1      # positive
            }
    meta_dat = dm.dataset_df.assign(
        y = lambda x: x[dm.target_disease].map(target_map)
    )

#%%
# Adding Race from processed demo data 
processed_demo = pd.read_csv("data/CheXpert/processed/cheXpert_processed_demo_data.csv")
df = (meta_dat
    .join(processed_demo.set_index("patient_id"), how = "left", on = "patient_id")
    .dropna(axis = 0, subset=processed_demo.columns)
)

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
desc_race.plot_positive_rate(title = f'Percentage with {disease}', orientation='h')
if save_figs: 
    plt.savefig(fig_path_report+f"posperc_race_{data_set}.pdf", bbox_inches='tight')
desc_race.plot_n_target_across_sens_var(
    orientation='h',
    return_ax=False, 
    **{"class_1_label":disease, "class_0_label": f"No {disease}"})
if save_figs: 
    plt.savefig(fig_path_report+f"N_race_{data_set}.pdf", bbox_inches='tight')

#%%
desc_race_sex = DescribeData(a_name = "race_sex", 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4, 
                        "specific_col_idx": [0, 4, 5, 6, 10, 11, 7, 9]})

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
