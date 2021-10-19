#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit

#%% Initialize parameters
figure_path = 'figures/evaluation_plots/'
fig_path_report = '../Thesis-report/00_figures/'

update_report_figures = False # Write new figures to report repository?

anym_w_fp = 0.2

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

# %% Saving plots for report 

# Confusion Matrix
FairKitDict["anym"].layer_3(method = 'confusion_matrix', output_table=False)
if update_report_figures:
    plt.savefig(f"{fig_path_report}confusion_matrix_anym.pdf", bbox_inches='tight')

# Positive weight influence
FairKitDict["anym"].layer_3(method = 'w_fp_influence', output_table=False)
if update_report_figures:
    plt.savefig(f"{fig_path_report}w_fp_influence_anym.pdf", bbox_inches='tight')

# Roc Curves 
FairKitDict["anym"].layer_3(method = 'roc_curves', output_table=False)
if update_report_figures:
    plt.savefig(f"{fig_path_report}roc_curve_anym.pdf", bbox_inches='tight')

# Calibration
FairKitDict["anym"].layer_3(method = 'calibration', output_table=False)
if update_report_figures:
    plt.savefig(f"{fig_path_report}calibration_anym.pdf", bbox_inches='tight')

# Independence Check
FairKitDict["anym"].layer_3(method = 'independence_check', output_table=False)
if update_report_figures:
    plt.savefig(f"{fig_path_report}independence_check_anym.pdf", bbox_inches='tight')

# %%
