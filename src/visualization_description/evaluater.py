#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit
from src.evaluation_tool.utils import static_split

#%% Initialize parameters
figure_path = 'figures/evaluation_plots/'
fig_path_report_l2 = '../Thesis-report/00_figures/evalL2/'
fig_path_report_l3 = '../Thesis-report/00_figures/evalL3/'

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
update_report_figures = True # Write new figures to report repository?
run_all_plots = False
run_l2_plots = False
run_l3_plots = False
run_anym_plots = True

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
    "Generate layer 1 overview table"
    l1_list = []
    for mod_name, kit in FairKitDict.items():
        l1 = kit.layer_1(plot = False)
        l1max = l1.loc[l1.weighted_misclassification_ratio == max(l1.weighted_misclassification_ratio)]
        
        acc = np.mean(kit.y == kit.y_hat)*100
        dataset_name, model_name = static_split(kit.model_name, ': ', 2)
        tab = pd.DataFrame({
            'Dataset': dataset_name,
            'Model': model_name,
            'N': kit.n,
            'w_fp': kit.w_fp,
            'Max WMQ': round(l1max.weighted_misclassification_ratio.iat[0], 1),
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
    # running all layers if nothing is specified in kwargs 
    if all(key not in kwargs.keys() for key in ["run_layer_1", 'run_layer_2', 'run_layer_3']):
        run_all = True 
    else: 
        run_all = False 

    if save_plots and plot_path is None:
        raise ValueError('You must supply a `plot_path` when `save_plots` = True')

    if kwargs.get("run_layer_1") or run_all:
        kit.layer_1(output_table = False)
        if save_plots: 
            plt.savefig(plot_path+'l1'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_layer_2") or run_all:
        kit.layer_2(output_table = False, **{"suptitle":True})
        if save_plots: 
            plt.savefig(plot_path+'l2'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_layer_3") or run_all:
        method_options = [
            'w_fp_influence', 'roc_curves', 'calibration', 
            'confusion_matrix', 'independence_check']
        for method in method_options:
            kit.layer_3(method = method, output_table = False)
            if save_plots: 
                path = plot_path+'l3_'+method+ext
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')
                plt.close()

        
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
                **{"run_layer_2":True})
    
    if run_l3_plots:
        for dataset, method in l3_report_plots:
            FairKitDict[dataset].layer_3(method = method, **{"cm_print_n":True})
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
            **{"run_layer_2":True})
        make_all_plots(kit, 
            save_plots = update_report_figures,
            plot_path = path,
            ext = ".pdf",
            **{"run_layer_3":True})

