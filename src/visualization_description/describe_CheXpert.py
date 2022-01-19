#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from src.visualization_description.descriptive_tool import DescribeData
from src.models.data_modules import CheXpertDataModule

fig_path_report = '../Thesis-report/00_figures/cheXpert/'
save_figs = False
disease = 'Cardiomegaly'

def get_descriptive_cheXpert_data(data_set):
    # Loading CheXpert
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

    # Adding Race from processed demo data 
    processed_demo = pd.read_csv("data/CheXpert/processed/cheXpert_processed_demo_data.csv")
    df = (meta_dat
        .join(processed_demo.set_index("patient_id"), how = "left", on = "patient_id")
        .dropna(axis = 0, subset=processed_demo.columns)
        .rename(mapper = str.lower, axis = 'columns')
    )

    return df

def get_figures_and_tables(
        df, a_name, disease = 'Cardiomegaly', fig_path_report = None, save_figs = False, orientation = 'h'):

    if fig_path_report is None:
        fig_path_report = '../Thesis-report/00_figures/cheXpert/'
    
    desc = DescribeData(a_name = a_name, 
                        y_name = "y", 
                        id_name = 'patient_id', 
                        data = df,
                        data_name=f'CheXpert, target: {disease}', 
                        **{"decimal":4})

    desc.descriptive_table_to_tex(target_tex_name=f'Has {disease}')
    desc.plot_positive_rate(title = f'Percentage with {disease}', orientation=orientation)
    if save_figs: 
        plt.savefig(fig_path_report+f"posperc_{a_name}_{data_set}.pdf", bbox_inches='tight')
    desc.plot_n_target_across_sens_var(
        orientation=orientation,
        return_ax=False, 
        **{"class_1_label":disease, "class_0_label": f"No {disease}"})
    if save_figs: 
        plt.savefig(fig_path_report+f"N_{a_name}_{data_set}.pdf", bbox_inches='tight')


#%%
if __name__ == "__main__":
    # Choose dataset: test, train or val
    data_set = "all"

    df = get_descriptive_cheXpert_data(data_set)

    # Get figures and tables for main report
    get_figures_and_tables(df, a_name = 'sex', orientation = 'v', save_figs = save_figs)
    get_figures_and_tables(df, a_name = 'race', save_figs = save_figs)
    get_figures_and_tables(df, a_name = 'race_sex', save_figs = save_figs)

    #%% Get prevalences in test/val/training
    # Train
    tbl_combined = []

    for data_set in ['train', 'test']:
        df = get_descriptive_cheXpert_data(data_set = data_set)
        for attr in ['sex', 'race', 'race_sex']:
            desc = DescribeData(a_name = attr, 
                                y_name = "y", 
                                id_name = 'patient_id', 
                                data = df,
                                data_name=f'CheXpert, target: {disease}', 
                                **{"decimal":4})
            tbl = desc.descriptive_table.assign(
                dataset = data_set, 
                sens_grp = attr)
            tbl_combined.append(tbl)
    tbl_combined = pd.concat(tbl_combined)

    #%%
    tex_pos_rate = (lambda x :
        [f"{x.N_positive[i]} ({x.positive_frac[i]*100}%)" for i in range(x.shape[0])]
        )
    tex_conf_int = (lambda x :
        [f"[{x.conf_lwr[i]*100}%, {x.conf_upr[i]*100}%]" for i in range(x.shape[0])]
        )
    col_order = [
        'sens_grp', 'a', 'dataset', 
        'N', 'N_positive', 'positive_frac','conf_lwr', 'conf_upr']
    tmp = (tbl_combined[col_order]
                .reset_index(drop = True)
                .round({'positive_frac': 4, 'conf_lwr': 4, 'conf_upr': 4})
                .assign(
                    N_positive = tex_pos_rate, 
                    CI = tex_conf_int)
                .drop(columns = ["conf_lwr", "conf_upr", "positive_frac"])
                .sort_values(['sens_grp', 'a', 'dataset'])
                .rename(columns = {
                    "a": 'Group', 
                    "N_positive": 'Has Cardiomegaly',
                    'sens_grp': 'Sensitive Attribute',
                    'dataset': 'Split'}))
    tmp.to_csv('references/split_distribution_for_tablesgenerator.csv')

# %%
