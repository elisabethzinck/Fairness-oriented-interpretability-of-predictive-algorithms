# imports 
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt

from src.visualization_description.descriptive_tool import DescribeData

update_report_figs = False
fig_path_report = '../Thesis-report/00_figures/describe_data/'

# reading data 
df = pd.read_csv('data/processed/taiwanese_credit.csv')

desc = DescribeData(data = df, 
                    y_name = "default_next_month",
                    a_name = 'sex', 
                    id_name = 'id', 
                    data_name = 'Taiwanese Credit Score')

desc.descriptive_table_to_tex(target_tex_name="Defaulted")
desc.plot_positive_rate(title = 'Percentage of Defaulters', orientation='v')

desc.plot_n_target_across_sens_var(
    orientation='v', 
    return_ax=True, 
    **{"class_0_label":'Not Defaulted', "class_1_label":'Defaulted'}
    )
if update_report_figs: 
    plt.savefig(fig_path_report+'taiwanese_N_by_sex.pdf', bbox_inches='tight')




