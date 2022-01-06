# imports 
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt

from src.visualization_description.descriptive_tool import DescribeData

update_report_figs = False
fig_path_report = '../Thesis-report/00_figures/describe_data/'

# reading data 
df = pd.read_csv("data/processed/compas/compas-scores-two-years-pred.csv")

desc = DescribeData(data = df, 
                    y_name = "two_year_recid",
                    a_name = 'race', 
                    id_name = 'id', 
                    data_name='COMPAS')

desc.descriptive_table_to_tex(target_tex_name="Recidivists")
desc.plot_positive_rate(title = 'Percentage of Recidivists', orientation='v')

desc.plot_n_target_across_sens_var(
    orientation='v', 
    return_ax=True, 
    **{"class_0_label":'Not Recidivated', "class_1_label":'Recidivated'}
    )
if update_report_figs: 
    plt.savefig(fig_path_report+'compas_N_by_race.pdf', bbox_inches='tight')




