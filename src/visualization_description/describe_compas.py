# imports 
import pandas as pd 
import numpy as np 
import os

# reading data 
df = pd.read_csv("data/processed/compas/compas-scores-two-years-pred.csv")

#%% create table for TeX

# Names and variables to be in table 
sens_var = "race"
target = "two_year_recid"
target_tex_name = "Recidivated (%)"

# Creating grouped table  
N_pos_func = lambda x: np.count_nonzero(x)
pos_perc_func = lambda x: (np.count_nonzero(x)/len(x))*100
N_pos_tab_func = lambda x: f"{N_pos_func(x)} ({pos_perc_func(x):.0f})%"

df_grouped = (df.groupby([sens_var])
    .agg(N = (target, "count"),
         N_positive = (target, N_pos_tab_func))
         .reset_index()
)
df_grouped.rename(columns = {target: sens_var.capitalize(), 
                             "N_positive": target_tex_name})
# Styling the data frame 
bold_cols = {'selector': 'th.col_heading',
             'props': 'font-weight: bold;'}
mid_rule = {'selector': 'midrule', 'props': ':hline;'}
s = df_grouped.style.format()
s.set_table_styles([bold_cols, mid_rule])

# printing style to Latex 
column_format =  "l"+"c"*(df_grouped.shape[1]-1)
s_tex = s.to_latex(column_format = column_format)
print(s_tex)


# %%


#%% 
header = ["Race", "N", "Recidivated"]

latex_table = df_grouped.to_latex(
    header = header,
    float_format = "%.0f",
    column_format = "l"+"c"*(df_grouped.shape[1]-1), 
    index = False,
    )

print(latex_table)


