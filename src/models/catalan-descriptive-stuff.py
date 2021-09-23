#%%
# imports 
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

from src.evaluation_tool.utils import custom_palette, abs_percentage_tick

file_path = 'data\\processed\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-subset.xlsx'
raw_data = pd.read_excel(file_path, index_col=0)


# %% how many individuals are there from each country? 
df1 = (raw_data.groupby(['V3_nationality_country'])
            .agg(N_people = ('id', 'count'))
            .reset_index()
            )

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x = 'N_people', data = df1)

# Grouping by N_people and adding a column with list of countries with that amount of N_people
df2 = (df1.groupby(['N_people'])
        .agg(N_countries = ('V3_nationality_country', 'count'))
        .reset_index()
        )
countries = []
for i in df2.N_people: 
    countries.append(df1[df1['N_people'] == i].V3_nationality_country.to_list())
df2['countries'] = countries


#%% How do they distribute across V4_area_origin? 
df_origin = (raw_data.groupby(['V4_area_origin'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)
bar1 = sns.barplot(
    x = 'N_people', y = 'V4_area_origin', 
    data = df_origin,
    estimator=sum,
    palette = custom_palette(specific_col_idx = [6]),
    label = 'Not Recidivated',
    ax = ax)
bar2 = sns.barplot(
    x = 'recidivated', y = 'V4_area_origin', 
    data = df_origin,
    palette = custom_palette(specific_col_idx = [2]),
    label ='Recidivated',
    ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('#Individuals by Area of Origin')
plt.legend()

# %% How do they distribute across age?
df_age = (raw_data.groupby(['V8_age'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)
bar1 = sns.barplot(
    x = 'V8_age', y = 'N_people', 
    data = df_age,
    estimator=sum,
    palette = custom_palette(specific_col_idx = [6]),
    label = 'Not Recidivated',
    ax = ax)
bar2 = sns.barplot(
    x = 'V8_age', y = 'recidivated', 
    data = df_age,
    palette = custom_palette(specific_col_idx = [2]),
    label ='Recidivated',
    ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('#Individuals by Age at Crime Time')
plt.legend()

# %% How do they distribute across crimes 
df_crime = (raw_data.groupby(['V14_main_crime', 'V15_main_crime_cat'])
            .agg(N_people = ('id', 'count'))
            .reset_index()
            .sort_values('V15_main_crime_cat', ascending=False)
        )

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x = 'N_people', data = df_crime)

#df_crime[df_crime.N_people > 30]

# %%
