#%%
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt 

from biasbalancer.descriptive_tool import DescribeData
from src.models.data_modules import ADNIDataModule

write_sankey = False
fig_path_report = '../Thesis-report/00_figures/describe_data/'
update_report_figs = False

#%%
adni_no = 2
dm = ADNIDataModule(dataset = adni_no)

# %% Create node labels
time_frames = ['label', '1y', '2y', '3y', '4y', '5y']
labels = [0,1,2,3]
nodes = []
for t in time_frames:
    nodes = nodes + [t+': '+str(lab) for lab in labels]

# %% Create links
n_flows = len(time_frames)-1
links = [None]*n_flows
mci = dm.raw_data['mci']
for i in range(n_flows):
    source = time_frames[i]
    target = time_frames[i+1]
    print(f's: {source}, t:{target}')
    links[i] = pd.DataFrame(
        {'source': [source + ': '+str(state) for state in mci[source]],
        'target': [target + ': '+str(state) for state in mci[target]]}
    )

links = (pd.concat(links)
    .groupby(['source', 'target'])
    .size()
    .reset_index(name = 'value'))


# %% Map to indexes instead of names
name_map = {name:idx for (name,idx) in zip(nodes, np.arange(len(nodes)))}
links['source'] = links['source'].map(name_map)
links['target'] = links['target'].map(name_map)


# %%
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = nodes,
      color = "blue"
    ),
    link = dict(
      source = links.source,
      target = links.target,
      value = links.value
  ))])
fig.show()
# %%
if write_sankey: 
    fig.write_image(f'figures/descriptive_plots/ADNI{adni_no}_sankey.png')

# %%

# Extracting data from data module
processed_data = (
    pd.concat([dm.processed_data['test_data'], dm.processed_data['trainval_data']])
    .reset_index(drop = False)
)

desc = DescribeData(y_name='y', 
                    a_name = 'sex',
                    id_name = 'rid',
                    data = processed_data, 
                    data_name = f'ADNI{adni_no}')

desc.plot_n_target_across_sens_var(
    orientation='v',
    return_ax=True, 
    **{"class_0_label":"No Alzheimer", "class_1_label":"Alzheimer"})
if update_report_figs: 
    plt.savefig(fig_path_report+f'ADNI{adni_no}_N_by_sex.pdf', bbox_inches='tight')

desc.descriptive_table_to_tex(target_tex_name='Has Alzheimer')


# %%

