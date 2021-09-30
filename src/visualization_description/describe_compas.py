# imports 
import pandas as pd 
import numpy as np 
import os

from pandas.core.indexes.range import RangeIndex

# reading data 
df = pd.read_csv("data/processed/compas/compas-scores-two-years-pred.csv")

#%% create table for TeX

# Names and variables to be in table 
sens_var = "race"
target = "two_year_recid"
target_tex_name = "Recidivated"

# Creating grouped table  
N_pos_func = lambda x: np.count_nonzero(x)
pos_perc_func = lambda x: (np.count_nonzero(x)/len(x))*100
N_pos_tab_func = lambda x: f"{N_pos_func(x)} ({pos_perc_func(x):.0f})%"

df_grouped = (df.groupby([sens_var])
    .agg(N = (target, "count"),
         N_positive = (target, N_pos_tab_func))
         .reset_index()
)

# appending total row: 
row_total = pd.DataFrame({
    sens_var: 'All', 
    "N": len(df[target]),
    "N_positive": N_pos_tab_func(df[target]),
    }, index = [0])

df_grouped =df_grouped.append(row_total, ignore_index=True)

df_grouped.rename(columns = {sens_var: sens_var.capitalize(), 
                             "N_positive": target_tex_name},
                  inplace = True)

# Styling the data frame 

mid_rule = {'selector': 'midrule', 'props': ':hline;'}
s = df_grouped.style.format(escape = "latex")
s.hide_index()
s.set_table_styles([mid_rule])

# printing style to Latex 
column_format =  "l"+"c"*(df_grouped.shape[1]-1)
s_tex = s.to_latex(column_format = column_format,
                   convert_css = True)
print(s_tex)

# %%



