#%% imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from biasbalancer.plots import custom_palette
from src.visualization_description.descriptive_tool import DescribeData

file_path = 'data\\processed\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-subset.csv'
raw_data = pd.read_csv(file_path, index_col=0).reset_index(drop = False)

fig_path_report = '../Thesis-report/00_figures/describe_data/'
update_report_figs = False

#%% Aggregated tables 
desc = DescribeData(a_name = "V4_area_origin", 
                    y_name = "V115_RECID2015_recid", 
                    id_name = 'id', 
                    data = raw_data,
                    data_name='Catalan Juvenile Recidivism')

desc.descriptive_table_to_tex(target_tex_name='Recidivists')
desc.plot_positive_rate(title = 'Percentage of Recidivists', orientation='h')
desc.plot_n_target_across_sens_var(
    orientation='h',
    return_ax=False, 
    **{"class_0_label":"Not Recidivated", "class_1_label":"Recidivated"})
if update_report_figs: 
    plt.savefig(fig_path_report+'catalan_N_by_area.pdf', bbox_inches='tight')
