# Descriptive stuff 

#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from seaborn.axisgrid import FacetGrid 

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

from src.data.general_preprocess_functions import one_hot_encode_mixed_data

#%% 
def get_fraction_of_group(group):
    """Helper function to calculate fraction of positives and negatives in
    each group. To be used in apply with groupby"""
    group['fraction_of_grp_obs'] = group['n'].agg(lambda x: x/x.sum())
    return group

class DescribeData:
    def __init__(self, y_name, a_name, data = None):
        """Saves and calculates all necessary attributes for FairKit object
        
        Args:
            y (string): Name of target variable
            a (string): Name of sensitive variable 
            data (data frame): Data frame with data. Defaults to None.

        """
        self.y_name = y_name
        self.a_name = a_name
        
        self.data = data 
        self.data = self.data.rename(columns = {self.y_name: 'y', self.a_name: 'a'})
        
        self.sens_grps = self.data.a.unique()
        self.n_sens_grps = len(self.sens_grps)
        self.group_n = self.data.groupby('a').size()

        # Define color palette
        cols = sns.color_palette(n_colors = self.n_sens_grps)
        self.sens_grps_cols = dict(zip(self.sens_grps, cols))

        self.grp_data = self.group_data()

    def group_data(self):
        if hasattr(self, 'rel_rates'):
            return self.rel_rates
        else:
            grp_data = (self.data.groupby(by=['a', 'y'])
                                  .size()
                                  .reset_index(name = 'n')
                                  .groupby(by='a')
                                  .apply(get_fraction_of_group)
                    )
            self.grp_data = grp_data
            return self.grp_data

    def plot_fraction_of_target(self):
        fig = plt.figure(figsize = (4,4))
        gs = GridSpec(nrows = 1, ncols = 1)
        ax = fig.add_subplot(gs[0,0])
        sns.barplot(x='y',
                    y='fraction_of_grp_obs',
                    hue = 'a',
                    data = self.grp_data, 
                    ax = ax)
        ax.set_ylim((0,1))
        handles, labels = ax.get_legend_handles_labels()
        labels = [f"{labels[i]} (N={self.group_n[i]})" for i in range(self.n_sens_grps)]
        ax.legend(handles, labels, frameon = False,
                  fontsize = 12, title = self.a_name, title_fontsize = 12)
        ax.set_ylim((0,1))
        ax.set_ylabel('Fraction of Observations', fontsize = 12)
        ax.set_xlabel(self.y_name, fontsize = 12)
        for pos in ['right', 'top', 'left']:
                ax.spines[pos].set_visible(False)
        ax.tick_params(left=False, labelsize=12)


#%%

if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit.csv'

    data = pd.read_csv(file_path)

    desc = DescribeData(y_name='credit_score', 
                        a_name = 'sex',
                        data = data)


    # PCA 
    X = (one_hot_encode_mixed_data(data)
          .drop(columns=['person_id', 'credit_score'])
          )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = 2
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    scores = pca.transform(X)

    

    # TODO plot pca and find out what the components are 

    

# %%
