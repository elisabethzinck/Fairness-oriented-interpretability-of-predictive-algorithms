#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from src.evaluation_tool.layered_tool import FairKit
from src.evaluation_tool.utils import static_split

#%% Initialize parameters
figure_path = 'figures/evaluation_plots/'
fig_path_report_l2 = '../Thesis-report/00_figures/evalL2/'
fig_path_report_l3 = '../Thesis-report/00_figures/evalL3/'
fig_path_chexpert = '../Thesis-report/00_figures/cheXpert/'

l3_report_plots = [
    ['german_logreg', 'confusion_matrix'],
    ['german_nn', 'confusion_matrix'],
    ['taiwanese_logreg', 'roc_curves'],
    ['taiwanese_nn', 'roc_curves'],
    ['catalan_logreg', 'roc_curves'],
    ['catalan_logreg', 'independence_check'],
    ['catalan_logreg', 'w_fp_influence'],
    ['adni2_nn', 'calibration']
]

update_figures  = False
update_report_figures = False # Write new figures to report repository?
run_all_plots = False
run_l2_plots = False
run_l3_plots = False
run_anym_plots = False
run_chexpert = True

make_cheXpert_table = True

#############################################
#%% Load data and initialize FairKit
##############################################
def get_FairKitDict(include_anym = True, include_ADNI = False):
    "Initialize all FairKits and save them in dict"
    FairKitDict = {}
    credit_w_fp = 0.9
    compas_w_fp = 0.9
    catalan_w_fp = 0.9
    anym_w_fp = 0.2
    adni_w_fp = 0.1

    # Anym
    if include_anym:
        anym = pd.read_csv('data/processed/anonymous_data.csv')
        FairKitDict['anym'] = FairKit(
            data = anym,
            y_name = 'y', 
            y_hat_name = 'yhat', 
            a_name = 'grp', 
            r_name = 'phat',
            w_fp = anym_w_fp,
            model_name = 'Anonymous Data')

    # German logistic regression
    german_log_reg = pd.read_csv('data/predictions/german_credit_log_reg.csv')
    FairKitDict['german_logreg'] = FairKit(
        data = german_log_reg,
        y_name = 'credit_score', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'sex', 
        r_name = 'log_reg_prob',
        w_fp = credit_w_fp,
        model_name = 'German Credit: Logistic regression')

    # German Neural network
    german_nn = pd.read_csv('data/predictions/german_credit_nn_pred.csv')
    FairKitDict['german_nn'] = FairKit(
        data = german_nn,
        y_name = 'credit_score', 
        y_hat_name = 'nn_pred', 
        a_name = 'sex', 
        r_name = 'nn_prob',
        w_fp = credit_w_fp,
        model_name = 'German Credit: Neural network')

    # Compas
    compas = (pd.read_csv('data/processed/compas/compas-scores-two-years-pred.csv').assign(scores = lambda x: x.decile_score/10))
    FairKitDict['compas'] = FairKit(
        data = compas,
        y_name = 'two_year_recid', 
        y_hat_name = 'pred', 
        a_name = 'race', 
        r_name = 'scores',
        w_fp = compas_w_fp,
        model_name = 'COMPAS: Decile scores')

    # Catalan Neural network
    catalan_logreg = pd.read_csv('data/predictions/catalan_log_reg.csv')
    FairKitDict['catalan_logreg'] = FairKit(
        data = catalan_logreg,
        y_name = 'V115_RECID2015_recid', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'V4_area_origin', 
        r_name = 'log_reg_prob',
        w_fp = catalan_w_fp,
        model_name = 'Catalan: Logistic regression')
    
    catalan_nn = pd.read_csv('data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv')
    FairKitDict['catalan_nn'] = FairKit(
        data = catalan_nn,
        y_name = 'V115_RECID2015_recid', 
        y_hat_name = 'nn_pred', 
        a_name = 'V4_area_origin', 
        r_name = 'nn_prob',
        w_fp = catalan_w_fp,
        model_name = 'Catalan: Neural network')

    # Taiwanese logreg
    taiwanese_logreg = pd.read_csv('data/predictions/taiwanese_log_reg.csv')
    FairKitDict['taiwanese_logreg'] = FairKit(
        data = taiwanese_logreg,
        y_name = 'default_next_month', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'sex', 
        r_name = 'log_reg_prob',
        w_fp = credit_w_fp,
        model_name = 'Taiwanese: Logistic regression')

    # Taiwanese nn
    taiwanese_nn = pd.read_csv('data/predictions/taiwanese_nn_pred.csv')
    FairKitDict['taiwanese_nn'] = FairKit(
        data = taiwanese_nn,
        y_name = 'default_next_month', 
        y_hat_name = 'nn_pred', 
        a_name = 'sex', 
        r_name = 'nn_prob',
        w_fp = credit_w_fp,
        model_name = 'Taiwanese: Neural network')


    if include_ADNI:
        for adni_no in [1,2]:
            adni = pd.read_csv(f'data/ADNI/predictions/ADNI{adni_no}_log_reg.csv')
            FairKitDict[f'adni{adni_no}_logreg'] = FairKit(
                data = adni,
                y_name = 'y', 
                y_hat_name = 'log_reg_pred', 
                a_name = 'sex', 
                r_name = 'log_reg_prob',
                w_fp = adni_w_fp,
                model_name = f'ADNI{adni_no}: Logistic regression')
            
            
            adni = pd.read_csv(f'data/ADNI/predictions/ADNI_{adni_no}_nn_pred.csv')
            FairKitDict[f'adni{adni_no}_nn'] = FairKit(
                data = adni,
                y_name = 'y', 
                y_hat_name = 'nn_pred', 
                a_name = 'sex', 
                r_name = 'nn_prob',
                w_fp = adni_w_fp,
                model_name = f'ADNI{adni_no}: Neural network')
    return FairKitDict
            
def get_l1_overview_table(print_latex = True):
    "Generate level 1 overview table"
    l1_list = []
    for kit in FairKitDict.values():
        l1 = kit.level_1(plot = False)
        l1max = l1.loc[l1.WMQ == max(l1.WMQ)]
        
        acc = np.mean(kit.y == kit.y_hat)*100
        dataset_name, model_name = static_split(kit.model_name, ': ', 2)
        tab = pd.DataFrame({
            'Dataset': dataset_name,
            'Model': model_name,
            'N': kit.n,
            'w_fp': kit.w_fp,
            'Max WMQ': round(l1max.WMQ.iat[0], 1),
            'Discriminated Group': l1max.grp.iat[0],
            'Accuracy': round(acc, 1)
        }, index = [0])
        l1_list.append(tab)
    l1tab = pd.concat(l1_list).query("Dataset != 'Anonymous Data'")
    if print_latex:
        print(l1tab.to_latex(index = False))
    
    return l1tab

def print_FairKitDict(FairKitDict):
    "Print all keys and model names in FairKitDict"
    for i, (mod_name, kit) in enumerate(FairKitDict.items()):
        print(f'{i} {mod_name}: {kit.model_name}') 

def make_all_plots(kit, save_plots = False, plot_path = None, ext = '.png', **kwargs):
    """ Makes all plots for FairKit instance kit

    Args:
        kit (FairKit): Object to plot figures from
        save_plots (bool): If true, plots are saved to `plot_path` and are not showed inline.
        plot_path (str): path to save plots in. Must be supplied if `save_plots` = True
        ext (str): Extension to use. Must begin with '.' (e.g. '.png')
    """
    # running all levels if nothing is specified in kwargs 
    if all(key not in kwargs.keys() for key in ["run_level_1", 'run_level_2', 'run_level_3']):
        run_all = True 
    else: 
        run_all = False 

    if save_plots and plot_path is None:
        raise ValueError('You must supply a `plot_path` when `save_plots` = True')

    if kwargs.get("run_level_1") or run_all:
        kit.level_1(output_table = False)
        if save_plots: 
            plt.savefig(plot_path+'l1'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_level_2") or run_all:
        kit.level_2(output_table = False, **{"suptitle":True})
        if save_plots: 
            plt.savefig(plot_path+'l2'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_level_3") or run_all:
        method_options = [
            'w_fp_influence', 'roc_curves', 'calibration', 
            'confusion_matrix', 'independence_check']
        for method in method_options:
            kit.level_3(method = method, output_table = False)
            if save_plots: 
                path = plot_path+'l3_'+method+ext
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')
                plt.close()

def get_chexpert_kits():
    """Creates dictionary of FairKits for CheXpert dataset"""
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
    
    df = (df.dropna(subset = ['gender', 'race'])
        .drop(columns = ['ethnicity'])
        .assign(y_hat = lambda x: x.scores >= 0.5)) # Fix mistake in pred
    
    # Initialize kits
    chexpert_kits = {}
    for sens_grp in ['gender', 'race', 'race_gender']:
        if sens_grp == 'race_gender':
            kit_df = df
            kwargs = {"specific_col_idx": [0, 4, 5, 6, 10, 11, 7, 9]}
        else:
            kit_df = df
            kwargs = {}

        mod_name = f'cheXpert_{sens_grp}'
        chexpert_kits[mod_name] = FairKit(
            data = kit_df,
            y_name = 'y',
            y_hat_name='y_hat',
            a_name = sens_grp,
            r_name = 'scores',
            w_fp = 0.1, 
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
    FairKitDict = get_FairKitDict() 
    print_FairKitDict(FairKitDict)

    l1tab = get_l1_overview_table()

    
    if run_all_plots: 
        # Make all(!) plots as png 
        for i, (mod_name, kit) in enumerate(FairKitDict.items()):
            print(i)
            path = figure_path + mod_name + '_'
            make_all_plots(kit, 
                save_plots = update_figures,
                plot_path = path)
        
    if run_l2_plots:
        for i, (mod_name, kit) in enumerate(FairKitDict.items()):
            print(i)
            if mod_name == "anym":
                continue
            path = fig_path_report_l2 + mod_name + '_'
            make_all_plots(kit, 
                save_plots = update_report_figures,
                plot_path = path,
                ext = ".pdf",
                **{"run_level_2":True})
    
    if run_l3_plots:
        for dataset, method in l3_report_plots:
            FairKitDict[dataset].level_3(method = method, **{"cm_print_n":True})
            if update_report_figures:
                path = fig_path_report_l3 + dataset + '_' + method + '.pdf'
                print(path)
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')

    if run_anym_plots:
        kit = FairKitDict['anym']
        path = '../Thesis-report/00_figures/anym/'
        make_all_plots(kit, 
            save_plots = update_report_figures,
            plot_path = path,
            ext = ".pdf",
            **{"run_level_2":True})
        make_all_plots(kit, 
            save_plots = update_report_figures,
            plot_path = path,
            ext = ".pdf",
            **{"run_level_3":True})


    if run_chexpert:
        chexpert_kits = get_chexpert_kits()

        if make_cheXpert_table:
            table_list = []
        for mod_name, kit in chexpert_kits.items():
            if run_all_plots:
                path = figure_path + mod_name + '_'
                make_all_plots(kit, 
                    save_plots = update_figures,
                    plot_path = path)

            if update_report_figures:
                path = fig_path_chexpert + mod_name + '_'
                make_all_plots(kit, 
                    save_plots = update_report_figures,
                    plot_path = path,
                    ext = ".pdf",
                    **{"run_level_2":True})
            
            if make_cheXpert_table:
                table_list.append(table_fairness_analysis(kit))

        if make_cheXpert_table:
            total_table = pd.concat(table_list)
            total_table.rename(
                {"sens_grp": "Sensitive Group",
                 "grp": "Subgroup",
                 "auc_roc": "AUC ROC",
                 "acc": "Accuracy %"         
                }
            ) 
            print(total_table.to_latex(index= False, float_format="%.3f"))
            


    # %%


# %%
