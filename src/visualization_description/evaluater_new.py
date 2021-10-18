#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit

#%% Initialize parameters
figure_path = 'figures/evaluation_plots/'
fig_path_report = '../Thesis-report/00_figures/'

run_all = False
update_figures  = False
update_report_figures = False # Write new figures to report repository?

credit_w_fp = 0.9
compas_w_fp = 0.9
catalan_w_fp = 0.9
anym_w_fp = 0.2
adni_w_fp = 0.1

#############################################
#%% Load data and initialize FairKit
##############################################
FairKitDict = {}

# Anym
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
    model_name = 'German Credit Logistic Regression')

# German neural network
german_nn = pd.read_csv('data/predictions/german_credit_nn_pred.csv')
FairKitDict['german_nn'] = FairKit(
    data = german_nn,
    y_name = 'credit_score', 
    y_hat_name = 'nn_pred', 
    a_name = 'sex', 
    r_name = 'nn_prob',
    w_fp = credit_w_fp,
    model_name = 'German Credit Neural Network')

# Compas
compas = (pd.read_csv('data/processed/compas/compas-scores-two-years-pred.csv')
    .query("race in ['African-American', 'Caucasian']")
    .assign(scores = lambda x: x.decile_score/10))
FairKitDict['compas'] = FairKit(
    data = compas,
    y_name = 'two_year_recid', 
    y_hat_name = 'pred_medium_high', 
    a_name = 'race', 
    r_name = 'scores',
    w_fp = compas_w_fp,
    model_name = 'COMPAS Decile Scores')

# Catalan neural network
catalan_nn = pd.read_csv('data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv')
FairKitDict['catalan_nn'] = FairKit(
    data = catalan_nn,
    y_name = 'V115_RECID2015_recid', 
    y_hat_name = 'nn_pred', 
    a_name = 'V4_area_origin', 
    r_name = 'nn_prob',
    w_fp = catalan_w_fp,
    model_name = 'Catalan Neural Network')

catalan_logreg = pd.read_csv('data/predictions/catalan_log_reg.csv')
FairKitDict['catalan_logreg'] = FairKit(
    data = catalan_logreg,
    y_name = 'V115_RECID2015_recid', 
    y_hat_name = 'log_reg_pred', 
    a_name = 'V4_area_origin', 
    r_name = 'log_reg_prob',
    w_fp = catalan_w_fp,
    model_name = 'Catalan Logistic Regression')

# Taiwanese nn
taiwanese_nn = pd.read_csv('data/predictions/taiwanese_nn_pred.csv')
FairKitDict['taiwanese_nn'] = FairKit(
    data = taiwanese_nn,
    y_name = 'default_next_month', 
    y_hat_name = 'nn_pred', 
    a_name = 'sex', 
    r_name = 'nn_prob',
    w_fp = credit_w_fp,
    model_name = 'Taiwanese Neural Network')

# Taiwanese logreg
taiwanese_logreg = pd.read_csv('data/predictions/taiwanese_log_reg.csv')
FairKitDict['taiwanese_logreg'] = FairKit(
    data = taiwanese_logreg,
    y_name = 'default_next_month', 
    y_hat_name = 'log_reg_pred', 
    a_name = 'sex', 
    r_name = 'log_reg_prob',
    w_fp = credit_w_fp,
    model_name = 'Taiwanese Logistic Regression')


for adni_no in [1,2]:
    adni = pd.read_csv(f'data/ADNI/predictions/ADNI_{adni_no}_nn_pred.csv')
    FairKitDict[f'adni{adni_no}_nn'] = FairKit(
        data = adni,
        y_name = 'y', 
        y_hat_name = 'nn_pred', 
        a_name = 'sex', 
        r_name = 'nn_prob',
        w_fp = adni_w_fp,
        model_name = f'ADNI{adni_no} Neural Network')
    
    adni = pd.read_csv(f'data/ADNI/predictions/ADNI{adni_no}_log_reg.csv')
    FairKitDict[f'adni{adni_no}_logreg'] = FairKit(
        data = adni,
        y_name = 'y', 
        y_hat_name = 'log_reg_pred', 
        a_name = 'sex', 
        r_name = 'log_reg_prob',
        w_fp = adni_w_fp,
        model_name = f'ADNI{adni_no} Logistic Regression')
            

# %% # Overview of models
for i, (mod_name, kit) in enumerate(FairKitDict.items()):
    print(f'{i} {mod_name}: {kit.model_name}')

# %% L1 overview table
l1_list = []
for mod_name, kit in FairKitDict.items():
    max_WMR = kit.layer_1(plot = False).weighted_misclassification_ratio.max()
    sens_grps_str = f"{{{', '.join(kit.sens_grps)}}}"
    acc = np.mean(kit.y == kit.y_hat)*100
    tab = pd.DataFrame({
        'dataset': kit.model_name,
        'n': kit.n,
        'sensitive groups': sens_grps_str,
        'max WMR': max_WMR,
        'acc': acc
    }, index = [0])
    l1_list.append(tab)
l1tab = pd.concat(l1_list)
l1tab