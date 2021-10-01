#%%
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from src.models.data_modules.ADNI_data_module import ADNIDataModule

#%%
dm = ADNIDataModule(dataset = 1)


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
fig.write_image('figures/descriptive_plots/ADNI_sankey.png')

# %%
