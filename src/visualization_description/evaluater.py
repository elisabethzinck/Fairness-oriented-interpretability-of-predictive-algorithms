#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from biasbalancer.balancer import BiasBalancer

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
    ['catalan_logreg', 'w_fp_influence']
]

update_figures  = False
update_report_figures = True # Write new figures to report repository?
run_all_plots = False
run_l2_plots = False
run_l3_plots = False
run_anym_plots = True

#############################################
#%% Load data and initialize BiasBalancer
##############################################
def get_BiasBalancerDict(include_anym = True):
    "Initialize all BiasBalancers and save them in dict"
    BiasBalancerDict = {}
    credit_w_fp = 0.9
    compas_w_fp = 0.9
    catalan_w_fp = 0.9
    anym_w_fp = 0.2

    # Anym
    if include_anym:
        anym = pd.read_csv('data/processed/anonymous_data.csv')
        BiasBalancerDict['anym'] = BiasBalancer(
            data = anym,
            y_name = 'y', 
            y_hat_name = 'yhat', 
            a_name = 'grp', 
            r_name = 'phat',
            w_fp = anym_w_fp,
            model_name = 'Example Data')

    # German logistic regression
    german_log_reg = pd.read_csv('data/predictions/german_credit_log_reg.csv')
    BiasBalancerDict['german_logreg'] = BiasBalancer(
        data = german_log_reg,
        y_name = 'credit_score', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'sex', 
        r_name = 'log_reg_prob',
        w_fp = credit_w_fp,
        model_name = 'German Credit: Logistic regression')

    # German Neural network
    german_nn = pd.read_csv('data/predictions/german_credit_nn_pred.csv')
    BiasBalancerDict['german_nn'] = BiasBalancer(
        data = german_nn,
        y_name = 'credit_score', 
        y_hat_name = 'nn_pred', 
        a_name = 'sex', 
        r_name = 'nn_prob',
        w_fp = credit_w_fp,
        model_name = 'German Credit: Neural network')

    # Compas
    compas = (pd.read_csv('data/processed/compas/compas-scores-two-years-pred.csv').assign(scores = lambda x: x.decile_score/10))
    BiasBalancerDict['compas'] = BiasBalancer(
        data = compas,
        y_name = 'two_year_recid', 
        y_hat_name = 'pred', 
        a_name = 'race', 
        r_name = 'scores',
        w_fp = compas_w_fp,
        model_name = 'COMPAS: Decile scores')

    # Catalan Neural network
    catalan_logreg = pd.read_csv('data/predictions/catalan_log_reg.csv')
    BiasBalancerDict['catalan_logreg'] = BiasBalancer(
        data = catalan_logreg,
        y_name = 'V115_RECID2015_recid', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'V4_area_origin', 
        r_name = 'log_reg_prob',
        w_fp = catalan_w_fp,
        model_name = 'Catalan: Logistic regression')
    
    catalan_nn = pd.read_csv('data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv')
    BiasBalancerDict['catalan_nn'] = BiasBalancer(
        data = catalan_nn,
        y_name = 'V115_RECID2015_recid', 
        y_hat_name = 'nn_pred', 
        a_name = 'V4_area_origin', 
        r_name = 'nn_prob',
        w_fp = catalan_w_fp,
        model_name = 'Catalan: Neural network')

    # Taiwanese logreg
    taiwanese_logreg = pd.read_csv('data/predictions/taiwanese_log_reg.csv')
    BiasBalancerDict['taiwanese_logreg'] = BiasBalancer(
        data = taiwanese_logreg,
        y_name = 'default_next_month', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'sex', 
        r_name = 'log_reg_prob',
        w_fp = credit_w_fp,
        model_name = 'Taiwanese: Logistic regression')

    # Taiwanese nn
    taiwanese_nn = pd.read_csv('data/predictions/taiwanese_nn_pred.csv')
    BiasBalancerDict['taiwanese_nn'] = BiasBalancer(
        data = taiwanese_nn,
        y_name = 'default_next_month', 
        y_hat_name = 'nn_pred', 
        a_name = 'sex', 
        r_name = 'nn_prob',
        w_fp = credit_w_fp,
        model_name = 'Taiwanese: Neural network')

    return BiasBalancerDict
            
def get_l1_overview_table(print_latex = True):
    
    def static_split(string, pattern, n_elem):
        "Same as str.split() except `n_elem`elements are always returned`"
        tmp = string.split(pattern, maxsplit = n_elem-1) # n_elem = n_split + 1
        n_remaining = n_elem - len(tmp)
        out = tmp + [None]*n_remaining
        return out

    "Generate level 1 overview table"
    l1_list = []
    for balancer in BiasBalancerDict.values():
        l1 = balancer.level_1(plot = False)
        l1max = l1.loc[l1.WMQ == max(l1.WMQ)]
        
        acc = np.mean(balancer.classifier.y == balancer.classifier.y_hat)*100
        dataset_name, model_name = static_split(balancer.model_name, ': ', 2)
        tab = pd.DataFrame({
            'Dataset': dataset_name,
            'Model': model_name,
            'N': balancer.n,
            'w_fp': balancer.w_fp,
            'Max WMQ': round(l1max.WMQ.iat[0], 1),
            'Discriminated Group': l1max.grp.iat[0],
            'Accuracy': round(acc, 1)
        }, index = [0])
        l1_list.append(tab)
    l1tab = pd.concat(l1_list).query("Dataset != 'Anonymous Data'")
    if print_latex:
        print(l1tab.to_latex(index = False))
    
    return l1tab

def print_BiasBalancerDict(BiasBalancerDict):
    "Print all keys and model names in BiasBalancerDict"
    for i, (mod_name, balancer) in enumerate(BiasBalancerDict.items()):
        print(f'{i} {mod_name}: {balancer.model_name}') 

def make_all_plots(balancer, save_plots = False, plot_path = None, ext = '.png', **kwargs):
    """ Makes all plots for BiasBalancer instance balancer

    Args:
        balancer (BiasBalancer): Object to plot figures from
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
        balancer.level_1(output_table = False)
        if save_plots: 
            plt.savefig(plot_path+'l1'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_level_2") or run_all:
        balancer.level_2(output_table = False)
        if save_plots: 
            plt.savefig(plot_path+'l2'+ext, bbox_inches='tight', facecolor = 'w')
            plt.close()

    if kwargs.get("run_level_3") or run_all:
        method_options = [
            'w_fp_influence', 'roc_curves', 'calibration', 
            'confusion_matrix', 'independence_check']
        for method in method_options:
            balancer.level_3(method = method, output_table = False, **{"cm_print_n":True})
            if save_plots: 
                path = plot_path+'l3_'+method+ext
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')
                plt.close()

        
#%%
if __name__ == '__main__':
    BiasBalancerDict = get_BiasBalancerDict() 
    print_BiasBalancerDict(BiasBalancerDict)

    l1tab = get_l1_overview_table()

    
    if run_all_plots: 
        # Make all(!) plots as png 
        for i, (mod_name, balancer) in enumerate(BiasBalancerDict.items()):
            print(i)
            path = figure_path + mod_name + '_'
            make_all_plots(balancer, 
                save_plots = update_figures,
                plot_path = path)
        
    if run_l2_plots:
        for i, (mod_name, balancer) in enumerate(BiasBalancerDict.items()):
            print(i)
            if mod_name == "anym":
                continue
            path = fig_path_report_l2 + mod_name + '_'
            make_all_plots(balancer, 
                save_plots = update_report_figures,
                plot_path = path,
                ext = ".pdf",
                **{"run_level_2":True})
    
    if run_l3_plots:
        for dataset, method in l3_report_plots:
            BiasBalancerDict[dataset].level_3(method = method, **{"cm_print_n":True})
            if update_report_figures:
                path = fig_path_report_l3 + dataset + '_' + method + '.pdf'
                print(path)
                plt.savefig(path, bbox_inches='tight', facecolor = 'w')

    if run_anym_plots:
        balancer = BiasBalancerDict['anym']
        path = '../Thesis-report/00_figures/anym/'
        make_all_plots(balancer, 
            save_plots = update_report_figures,
            plot_path = path,
            ext = ".pdf",
            **{"run_level_2":True})
        make_all_plots(balancer, 
            save_plots = update_report_figures,
            plot_path = path,
            ext = ".pdf",
            **{"run_level_3":True})

