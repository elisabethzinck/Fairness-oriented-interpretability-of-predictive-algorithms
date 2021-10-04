#%% imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from src.evaluation_tool.utils import custom_palette, abs_percentage_tick
from src.visualization_description.descriptive_functions import DescribeData

file_path = 'data\\processed\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-subset.csv'
raw_data = pd.read_csv(file_path, index_col=0)

fig_path_report = '../Thesis-report/00_figures/'
update_report_figs = True

#%% Aggregated tables 
desc_V2 = DescribeData(a_name = "V2_nationality_type", 
                    y_name = "V115_RECID2015_recid", 
                    id_name = 'id', 
                    data = raw_data)

desc_V4 = DescribeData(a_name = "V4_area_origin", 
                    y_name = "V115_RECID2015_recid", 
                    id_name = 'id', 
                    data = raw_data)

desc_V4.agg_table(to_latex=True, target_tex_name='Recidivists')



#%% How do they distribute across V4_area_origin? 
df_origin = (raw_data.groupby(['V4_area_origin'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )

fig = plt.figure(figsize=(5,3.5))
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
ax.set_xlabel('Number of Juveniles')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('Number of People by Area of Origin')
plt.legend()
if update_report_figs: 
    plt.savefig(fig_path_report+'V4_area_origin.pdf', bbox_inches='tight')



#%% How do they distribute across V2_nationality_type
df_foreign = (raw_data.groupby(['V2_nationality_type'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )

fig = plt.figure(figsize=(5,1.5))
ax = fig.add_subplot(1, 1, 1)
bar1 = sns.barplot(
    x = 'N_people', y = 'V2_nationality_type', 
    data = df_foreign,
    estimator=sum,
    palette = custom_palette(specific_col_idx = [6]),
    label = 'Not Recidivated',
    ax = ax)
bar2 = sns.barplot(
    x = 'recidivated', y = 'V2_nationality_type', 
    data = df_foreign,
    palette = custom_palette(specific_col_idx = [2]),
    label ='Recidivated',
    ax = ax)
ax.set_xlabel('Number of Juveniles')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('Number of People by Nationality Type')
plt.legend()
if update_report_figs: 
    plt.savefig(fig_path_report+'V2_nationality_type.pdf', bbox_inches='tight')


#%% How do they distribute across V1_sex? 
df_sex = (raw_data.groupby(['V1_sex'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(1, 1, 1)
bar1 = sns.barplot(
    y = 'N_people', x = 'V1_sex', 
    data = df_sex,
    estimator=sum,
    palette = custom_palette(specific_col_idx = [6]),
    label = 'Not Recidivated',
    ax = ax)
bar2 = sns.barplot(
    y = 'recidivated', x = 'V1_sex', 
    data = df_sex,
    palette = custom_palette(specific_col_idx = [2]),
    label ='Recidivated',
    ax = ax)
ax.set_xlabel('Number of Juveniles')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('#People by Sex')
plt.legend()

#%% How do they distribute across V6_province? 
df_province = (raw_data.groupby(['V6_province'])
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
    x = 'N_people', y = 'V6_province', 
    data = df_province,
    estimator=sum,
    palette = custom_palette(specific_col_idx = [6]),
    label = 'Not Recidivated',
    ax = ax)
bar2 = sns.barplot(
    x = 'recidivated', y = 'V6_province', 
    data = df_province,
    palette = custom_palette(specific_col_idx = [2]),
    label ='Recidivated',
    ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
for pos in ['right', 'top', 'left']:
    ax.spines[pos].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.set_title('#People by Province')
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

fig = plt.figure(figsize=(4,3))
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
ax.set_title('#People by Age at Crime Time')
ax.set_xlabel('Age in Years')
plt.legend()

# %% How do they distribute across crimes 
df_crime = (raw_data.groupby(['V19_committed_crime', 'V15_main_crime_cat'])
            .agg(N_people = ('id', 'count'))
            .reset_index()
            .sort_values('V19_committed_crime', ascending=False)
        )

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x = 'N_people', 
            y = 'V15_main_crime_cat',
            data = df_crime,
            ax = ax, 
            palette = custom_palette(n_colors = 3))
ax.set_ylabel('')
ax.set_title('#People pr. Crime Category in V19_committed_crime \n Partitioned by Main Crime Category')

#%% How do they distribute across V26
df_26 = (raw_data.groupby(['V26_finished_measure_grouped'])
                .agg(N_people = ('id', 'count'), 
                     recidivated = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)),
                     not_recidivated = ('V115_RECID2015_recid', lambda x: len(x)-np.count_nonzero(x)),
                     recid_frac = ('V115_RECID2015_recid', lambda x: np.count_nonzero(x)/len(x)))
                .reset_index()
                .sort_values(by = 'N_people', ascending = False)
            )
# %%