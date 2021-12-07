#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from src.evaluation_tool.layered_tool import FairKit
from src.visualization_description.evaluater import make_all_plots

from sklearn.metrics import roc_curve

#%% Initialize parameters
figure_path = 'figures/evaluation_plots/'
fig_path_chexpert = '../Thesis-report/00_figures/cheXpert/'

l3_report_plots_chexpert = [
    ['cheXpert_race', 'roc_curves'],
    ['cheXpert_race', 'calibration']] # Add more here later

update_figures  = True
update_report_figures = True # Write new figures to report repository?
run_all_plots = True
run_l3_plots = True
make_table = True

#############################################
#%% Functions
##############################################

def get_fpr_based_threshold():
    """Calculate threshold achieving fpr=0.2 on training data"""
    pred_path = 'data/CheXpert/predictions/adam_dp=2e-1/train_best_predictions.csv'
    demo_path = 'data/CheXpert/processed/cheXpert_processed_demo_data.csv'
    preds = pd.read_csv(pred_path)
    demo_data = pd.read_csv(demo_path)

    # Merge chexpert and demographic data
    df = preds.join(
        demo_data.set_index('patient_id'), 
        how = 'left', 
        on = 'patient_id')
    
    
    df = (df.dropna(subset = ['sex', 'race'])
        .drop(columns = ['ethnicity']))

    fpr_target = 0.2
    fpr, tpr, threshold = roc_curve(
            y_true = df.y, 
            y_score = df.scores)
    roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
    roc = (roc[roc.fpr < fpr_target]
        .sort_values('fpr', ascending = False)
        .reset_index(drop = True))
    tau = roc.threshold[0]
    return tau
    
def get_chexpert_prediction_data(threshold = 0.5):
    """Merges predictions with demodata and makes predictions based on threshold"""
    # Load data
    pred_path = 'data/CheXpert/predictions/adam_dp=2e-1/test_best_predictions.csv'
    demo_path = 'data/CheXpert/processed/cheXpert_processed_demo_data.csv'
    preds = pd.read_csv(pred_path)
    demo_data = pd.read_csv(demo_path)

    # Merge chexpert and demographic data
    df = preds.join(
        demo_data.set_index('patient_id'), 
        how = 'left', 
        on = 'patient_id')
    
    # 160 without any demographic information and 181 without ethnicity dropped
    assert df.race.isnull().sum() == 160 
    assert df.ethnicity.isnull().sum() == 181
    
    df = (df.dropna(subset = ['sex', 'race'])
        .drop(columns = ['ethnicity']))

    df = df.assign(y_hat = lambda x: x.scores >= threshold)
    
    return df

def get_chexpert_kits(pred_data):
    """Creates dictionary of FairKits for CheXpert dataset
    
    Params:
        pred_data (pd.DataFrame): Dataframe as returned by get_chexpert_prediction_data()
    """
    
    # Initialize kits
    chexpert_kits = {}
    for sens_grp in ['sex', 'race', 'race_sex']:
        if sens_grp == 'race_sex':
            kwargs = {"specific_col_idx": [0, 4, 5, 6, 10, 11, 7, 9]}
        else:
            kwargs = {}

        key_name = f'cheXpert_{sens_grp}'
        mod_name = f'CheXpert: {sens_grp}'
        chexpert_kits[key_name] = FairKit(
            data = pred_data,
            y_name = 'y',
            y_hat_name='y_hat',
            a_name = sens_grp,
            r_name = 'scores',
            w_fp = 0.1, 
            model_name = mod_name,
            **kwargs
        )
    return chexpert_kits

def table_fairness_analysis(fairKit_instance, to_latex = False):
    a_name = fairKit_instance.a_name
    
    # helper function
    def stats_group(df):
        scores = df.scores
        y_hat = df.y_hat
        y = df.y
        auc_roc = sklearn.metrics.roc_auc_score(y, scores)
        acc = sklearn.metrics.accuracy_score(y, y_hat)*100
        return [auc_roc, acc]

    auc_acc_table = (fairKit_instance.data
        .assign(roc_auc_tuple = lambda x: list(zip(x.y, x.scores)),
                acc_tuple = lambda x: list(zip(x.y, x.y_hat)))
        .groupby(a_name)
            .apply(stats_group)
        .apply(pd.Series)
        .reset_index()
        .rename(columns = {0:"auc_roc", 1:"acc"})
            )

    return_table = (fairKit_instance.level_1(plot = False)
        .rename(columns={"grp":a_name})
        .join(auc_acc_table.set_index(a_name), how = "left", on = a_name)
        .rename(columns={f"{a_name}":"grp"})
        .assign(sens_grp = a_name))

    return_table = return_table[["sens_grp","grp","n","WMR","WMQ","auc_roc","acc"]]

    if to_latex:
        print(return_table.to_latex(index = False, float_format="%.3f"))
    else:
        return return_table

        
#%%
if __name__ == '__main__':

    # initialize fairkits
    threshold = get_fpr_based_threshold()
    chexpert_df = get_chexpert_prediction_data(
        threshold = threshold)
    chexpert_kits = get_chexpert_kits(chexpert_df)

    if make_table:
        table_list = []
    for mod_name, kit in chexpert_kits.items():

        # Get all plots in png
        if run_all_plots:
            path = figure_path + mod_name + '_'
            make_all_plots(kit, 
                save_plots = update_figures,
                plot_path = path, 
                **{'suptitle': False, 'threshold': threshold})

        # Get plots for report
        if update_report_figures:
            path = fig_path_chexpert + mod_name + '_'
            make_all_plots(kit, 
                save_plots = update_report_figures,
                plot_path = path,
                ext = ".pdf",
                **{"run_level_2": True, 'suptitle': False})
        
        if make_table:
            table_list.append(table_fairness_analysis(kit))

    if make_table:
        total_table = pd.concat(table_list)
        total_table.rename(
            {"sens_grp": "Sensitive Group",
                "grp": "Subgroup",
                "auc_roc": "AUC ROC",
                "acc": "Accuracy %"         
            }
        ) 
        print(total_table.to_latex(index= False, float_format="%.3f"))
       
    if run_l3_plots:
        kwargs = {"threshold": threshold, "n_bins": 10}
        for dataset, method in l3_report_plots_chexpert:
            chexpert_kits[dataset].level_3(
                method = method, 
                **kwargs)
            if update_report_figures:
                path = fig_path_chexpert + dataset + '_' + method + '.pdf'
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')


# %%


