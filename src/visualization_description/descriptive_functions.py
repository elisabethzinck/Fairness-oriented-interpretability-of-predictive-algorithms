# Descriptive stuff 

#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from seaborn.axisgrid import FacetGrid 

from statsmodels.stats.proportion import proportion_confint

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.evaluation_tool.utils import custom_palette, error_bars, desaturate
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
        
        self.sens_grps = sorted(self.data.a.unique())
        self.n_sens_grps = len(self.sens_grps)
        self.group_n = self.data.groupby('a').size()

        # Define color palette
        self.sens_grps_cols = dict(
            zip(self.sens_grps, custom_palette(n_colors = self.n_sens_grps))
            )

        self.grp_data = self.group_data()
        self.agg_data = self.agg_table_2()

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


    def agg_table_2(self):
        """Positive Rates and confidence intervals of data 
        aggregated on the sensitive variable"""
        # helper lambda functions
        N_pos = lambda x: np.count_nonzero(x)
        pos_perc = lambda x: (np.count_nonzero(x)/len(x))
        confint = lambda x: proportion_confint(count=N_pos(x),
                                       nobs=len(x),
                                       method = "wilson")
        confint_lwr = lambda x: confint(x)[0]
        confint_upr = lambda x: confint(x)[1]
        # Creating grouped table  
        df_agg = (self.data.groupby(["a"])
            .agg(N = ("y", "count"),
                N_positive = ("y", N_pos),
                positive_frac = ("y", pos_perc), 
                conf_lwr = ("y", confint_lwr),
                conf_upr = ("y", confint_upr))
            .sort_values(by = 'N', ascending = False)
            .reset_index()
        )
        # appending total row: 
        row_total = pd.DataFrame({
            "a": 'Total', 
            "N": len(self.data.y),
            "N_positive": N_pos(self.data.y),
            "positive_frac": pos_perc(self.data.y),
            "conf_lwr": confint_lwr(self.data.y),
            "conf_upr": confint_upr(self.data.y)
            }, index = [self.n_sens_grps])
        df_agg =df_agg.append(row_total, ignore_index=True)

        return df_agg

    def agg_table_to_tex(self, target_tex_name = None):
        """Formats self.agg_data to a latex table"""
        if target_tex_name is None: 
            target_tex_name = self.y_name
        
        # helper lambda functions
        tex_pos_rate = (lambda x :
            [f"{x.N_positive[i]} ({x.positive_frac[i]*100:.0f}%)" for i in range(x.shape[0])]
            )
        tex_conf_int = (lambda x :
            [f"[{x.conf_lwr[i]*100:.0f}%, {x.conf_upr[i]*100:.0f}%]" for i in range(x.shape[0])]
            )

        agg_data_tex = (self.agg_data
            .assign(
                N_positive = tex_pos_rate, 
                CI = tex_conf_int)
            .drop(columns = ["conf_lwr", "conf_upr", "positive_frac"])
        )
        agg_data_tex.rename(columns = 
            {"a": self.a_name.capitalize().replace("_", "\_"), 
             "N_positive": target_tex_name},
             inplace = True)
        
        # Styling the data frame
        mid_rule = {'selector': 'midrule', 'props': ':hline;'}
        s = agg_data_tex.style.format(escape = "latex")
        s.hide_index()
        s.set_table_styles([mid_rule])
        column_format =  "lcr"
        s_tex = s.to_latex(column_format = column_format,
                        convert_css = False)
        
        # printing Tex code and returning df 
        print(s_tex)
        return s

    def plot_positive_rate(self, title =None):
        if title is None: 
            title = "Positive Fraction per Group"

        plot_df = (self.agg_data.query("a != 'Total'")
            .sort_values(by = 'a'))
        
        fig = plt.figure(figsize = (4,4))
        gs = GridSpec(nrows = 1, ncols = 1)
        ax = fig.add_subplot(gs[0,0])
        sns.barplot(y="positive_frac",
                    x = "a",
                    ax = ax,
                    data = plot_df,
                    order = self.sens_grps,
                    alpha = 0.95)
        ax.set_ylim((0,1))
        ax.set_ylabel('', fontsize = 12)
        ax.set_xlabel(self.a_name.capitalize(), fontsize = 12)
        sns.despine(ax = ax, top = True, right = True)
        ax.tick_params(left=False, labelsize=12)
        for bar, col in zip(ax.patches, self.sens_grps_cols.values()):
            bar.set_color(desaturate(col))

        error_bars(ax, data = plot_df)
        ax.set_title(title)
        ax.legend()
        ax.set_xticklabels([self.sens_grps[i].capitalize() for i in range(self.n_sens_grps)])

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
    #desc.agg_table_to_tex(target_tex_name='Defaulted')
    desc.plot_positive_rate('Fraction of Bad Credit Scores')
    #figure_path = 'figures/descriptive_plots/'
    #plt.savefig(figure_path+'tsne_sex.pdf', bbox_inches='tight')
   

# %%
