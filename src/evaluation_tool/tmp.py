#%%
from operator import index
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from matplotlib import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.evaluation_tool.utils import (custom_palette, desaturate)
from src.evaluation_tool.layered_tool import FairKit
#%%
def rainbow_text(x, y, strings, colors, ax=None, **kw):
    """
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

    The text will get added to the ``ax`` axes, if provided, otherwise the
    currently active axes will be used.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    # horizontal version
    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

    # vertical version
    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t,
                       rotation=90, va='bottom', ha='center', **kw)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, y=ex.height, units='dots')

#%%
file_path = 'data\\predictions\\german_credit_log_reg.csv'
german_log_reg = pd.read_csv(file_path)

fair_german_log_reg = FairKit(
    y = german_log_reg.credit_score, 
    y_hat = german_log_reg.log_reg_pred, 
    a = german_log_reg.sex, 
    r = german_log_reg.log_reg_prob,
    w_fp = 0.9,
    model_type='Logistic Regression')


#%%
df = fair_german_log_reg.layer_1(output_table=True)
max_idx = df.weighted_misclassification_ratio.idxmax() 
max_grp = df.grp[max_idx]
max_WMR = df.weighted_misclassification_ratio[max_idx]
max_color = desaturate(fair_german_log_reg.sens_grps_cols[max_grp])

line_1 = (f"The weighted misclassification rate of").split()
line_2 = (f"{max_grp.capitalize()} is {max_WMR:.0f}% larger").split()
line_3 = (f"than the smallest weigthed misclassification rate").split()


# Formatting line 2 
n_words_2 = len(line_2)
color_list = ['k']*n_words_2 #(58/255, 58/255, 58/255)
font_sizes = [20]*n_words_2

num_idx = 0
grp_idx = 2

color_list[grp_idx] = max_color
font_sizes[grp_idx] = 30
font_sizes[num_idx] = 30

#%%
fig = plt.figure(figsize = (6,2))
gs = GridSpec(nrows = 1, ncols = 1)
ax = fig.add_subplot(gs[0,0])

x = 0
y = 0

if ax is None:
    ax = plt.gca()


def format_text(ax, x, y, text_list, color_list, font_sizes):
    t = ax.transData
    canvas = ax.figure.canvas
    # horizontal version
    for s, c, f in zip(text_list, color_list, font_sizes):
        text = ax.text(x, y, s + " ", color=c, fontsize = f, transform=t)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')



#%%
fig = plt.figure(figsize = (6,2))
gs = GridSpec(nrows = 1, ncols = 1)
ax = fig.add_subplot(gs[0,0])
ax.set_xlim(0,9)
ax.set_ylim(0,3)
sns.despine(top = True, bottom = True, left = True, right= True)



ax.text(0.25,2.5,"The", fontsize = 25)
ax.text(1.65,2.5,max_grp.capitalize(), fontsize = 25, color = max_color)




rainbow_text(0, 0, "all unicorns poop rainbows ! ! !".split(),
             ['red', 'cyan', 'brown', 'green', 'blue', 'purple', 'black'],
             size=16)

plt.show()# %%

# %%
