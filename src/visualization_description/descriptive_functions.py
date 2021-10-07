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
from sklearn.manifold import TSNE

from src.data.general_preprocess_functions import one_hot_encode_mixed_data

#%% 
def get_fraction_of_group(group):
    """Helper function to calculate fraction of positives and negatives in
    each group. To be used in apply with groupby"""
    group['fraction_of_grp_obs'] = group['n'].agg(lambda x: x/x.sum())
    return group

class DescribeData:
    def __init__(self, y_name, a_name, id_name = None, data = None):
        """Saves and calculates all necessary attributes for FairKit object
        
        Args:
            y_name (string): Name of target variable
            a_name (string): Name of sensitive variable 
            id_name (string): Name of id variable
            data (data frame): Data frame with data

        """
        self.y_name = y_name
        self.a_name = a_name
        self.id_name = id_name
        
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


    def plot_tSNE(self, n_tries = None, perplexity = None, verbose = True):
        """Plot tSNE hightlighting sensitive groups.
        
        Args:
            n_tries (int): Number of perplexity values to try 
            perplexity (int): Perplexity parameter for tSNE
            verbose (bool): Control verbosity of method

        Either n_tries or perplexity must be supplied. If both are supplied, the method raises an error.

        """
        
        # Input checks
        if perplexity is not None and n_tries is not None:
            raise ValueError('You cannot specify both perplexity and n_tries')
        if perplexity is None and n_tries is None:
            raise ValueError('You must specify either perplexity or n_tries')
        
        # Prepare data
        if self.id_name is not None:
            cols_to_drop = [self.id_name, 'y', 'a']
        else:
            cols_to_drop = ['y', 'a']
        X = self.data.drop(columns=cols_to_drop)
        X = one_hot_encode_mixed_data(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate embedding
        if perplexity is not None: 
            tsne = TSNE(n_components = 2, perplexity = perplexity, init = 'pca', random_state = 42)
            X_tSNE_best = tsne.fit_transform(X_scaled)
        else:
            # orig paper says perplexities between 5 and 50 are reasonable
            perplexities = np.linspace(5,50, n_tries) 
            kl_min = np.inf # We want to minimize kl divergence

            for i, perp in enumerate(perplexities):
                tsne = TSNE(n_components = 2, perplexity = perp, init = 'pca', random_state = 42)
                X_tSNE_cur = tsne.fit_transform(X_scaled)
                kl_cur = tsne.kl_divergence_

                if kl_cur < kl_min:
                    kl_min = kl_cur
                    X_tSNE_best = X_tSNE_cur

                if verbose:
                    print(f'it {i+1}/{len(perplexities)}: \t perplexity: {perp} \t current fit: {kl_cur:.2f}')
        # Plot embedding
        plt.clf() # Avoid continuing on previous plot
        plotdf = self.data.rename(
            columns = {'y': self.y_name, 'a': self.a_name})
        sns.scatterplot(
            x= X_tSNE_best[:,0], 
            y = X_tSNE_best[:,1],
            hue = plotdf[self.a_name], 
            style= plotdf[self.y_name])
        plt.legend(
            bbox_to_anchor = (1,1), 
            loc = 'upper left') # Legend outside top right
        plt.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False)
        plt.title('t-SNE of Data')

    def agg_table(self, target_tex_name = None, to_latex = False):
        if target_tex_name is None: 
            target_tex_name = self.y_name
        
        # helper lambda functions
        N_pos_func = lambda x: np.count_nonzero(x)
        pos_perc_func = lambda x: (np.count_nonzero(x)/len(x))*100
        N_pos_tab_func = lambda x: f"{N_pos_func(x)} ({pos_perc_func(x):.0f}%)"
        
        # Creating grouped table  
        df_grouped = (self.data.groupby(["a"])
           .agg(N = ("y", "count"),
                N_positive = ("y", N_pos_tab_func))
            .reset_index()
            .sort_values(by = 'N', ascending = False)
        )

        # appending total row: 
        row_total = pd.DataFrame({
            "a": 'All', 
            "N": len(self.data.y),
            "N_positive": N_pos_tab_func(self.data.y),
            }, index = [0])

        df_grouped =df_grouped.append(row_total, ignore_index=True)

        df_grouped.rename(columns = {"a": self.a_name.capitalize().replace("_", "\_"), 
                                    "N_positive": target_tex_name},
                        inplace = True)
        
        if to_latex: 
            # Styling the data frame 
            mid_rule = {'selector': 'midrule', 'props': ':hline;'}
            s = df_grouped.style.format(escape = "latex")
            s.hide_index()
            s.set_table_styles([mid_rule])

            # printing style to Latex 
            column_format =  "lcr"
            s_tex = s.to_latex(column_format = column_format,
                            convert_css = False)
            print(s_tex)
            return
        else: 
            return df_grouped


#%%

if __name__ == "__main__":

    # German
    file_path = 'data\\processed\\german_credit_full.csv'

    data = pd.read_csv(file_path)

    desc = DescribeData(y_name='credit_score', 
                        a_name = 'sex',
                        id_name = 'person_id',
                        data = data)

    #desc.plot_tSNE(n_tries = 3)
    desc.plot_fraction_of_target()
    #figure_path = 'figures/descriptive_plots/'
    #plt.savefig(figure_path+'tsne_sex.pdf', bbox_inches='tight')
   

# %%

# Creating positive proportion plot w. confidence interval. 
import statsmodels

(data.groupby(['sex'])
    .agg(N_pos = ("credit_score", lambda x: np.count_nonzero(x)),
    N = ("person_id", "count"))
)

# %%
