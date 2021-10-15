#%%
from operator import index
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.ticker as mtick

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.evaluation_tool.utils import (
    custom_palette, error_bar, 
    N_pos, frac_pos, wilson_confint, error_bar, abs_percentage_tick)
#%% 
class DescribeData:
    def __init__(self, y_name, a_name, id_name = None, data = None):
        """Methods to describe data with focus on sensitive groups
        
        Args:
            y_name (string): Name of target variable
            a_name (string): Name of sensitive variable 
            id_name (string): Name of id variable
            data (data frame): Data frame with data

        """
        self.y_name = y_name
        self.a_name = a_name
        self.id_name = id_name
        
        self.data = data.rename(columns = {self.y_name: 'y', self.a_name: 'a'})
        
        self.sens_grps = sorted(self.data.a.unique())
        self.n_sens_grps = len(self.sens_grps)

        self.sens_grps_cols = dict(
            zip(self.sens_grps, custom_palette(n_colors = self.n_sens_grps))
            )
        self.descriptive_table = self.get_descriptive_table()

    def get_descriptive_table(self):
        """Positive rates and confidence intervals of data 
        aggregated by the sensitive variable"""
        df_agg = (self.data
            .groupby(["a"], as_index = False)
            .agg(
                N = ("y", "count"),
                N_positive = ("y", N_pos),
                positive_frac = ("y", frac_pos))
            .assign(
                conf_lwr = lambda x: wilson_confint(x.N_positive, x.N, 'lwr'),
                conf_upr = lambda x: wilson_confint(x.N_positive, x.N, 'upr'))
            .sort_values(by = 'N', ascending = False)
        )
        return df_agg

    def descriptive_table_to_tex(self, target_tex_name = None):
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
        
        # appending total row: 
        row_total = (
            pd.DataFrame({
                "a": 'Total', 
                "N": len(self.data.y),
                "N_positive": N_pos(self.data.y),
                "positive_frac": frac_pos(self.data.y)}, 
                index = [self.n_sens_grps])
            .assign(
                conf_lwr = lambda x: wilson_confint(x.N_positive, x.N, 'lwr'),
                conf_upr = lambda x: wilson_confint(x.N_positive, x.N, 'upr')))
        df_agg =self.descriptive_table.append(row_total, ignore_index=True)

        # Latex formatting
        agg_data_tex = (df_agg
            .assign(
                N_positive = tex_pos_rate, 
                CI = tex_conf_int)
            .drop(columns = ["conf_lwr", "conf_upr", "positive_frac"])
            .rename(columns = {
                "a": self.a_name.capitalize().replace("_", "\_"), 
                "N_positive": target_tex_name}))
        
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

    def plot_positive_rate(self, orientation = 'v', title = "Positive Fraction per Group"):
        assert orientation in ["v", "h"], "Choose orientation 'v' or 'h'"

        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(1,1,1)

        # Convert into percentage 
        plot_data = (self.descriptive_table
            .assign(positive_perc = lambda x: x.positive_frac*100,
                    conf_lwr = lambda x: x.conf_lwr*100,
                    conf_upr = lambda x: x.conf_upr*100)
        )

        for grp in self.sens_grps:
            plot_df = plot_data.query(f'a == "{grp}"')
            assert plot_df.shape[0] == 1
            bar_mid = plot_df.index[0]
            if orientation == 'v':
                sns.barplot(y="positive_perc",
                        x = "a",
                        ax = ax,
                        data = plot_df,
                        color = self.sens_grps_cols[grp], 
                        order = self.sens_grps,
                        alpha = 0.95)
                ax.set_ylim((0,100))
                ax.set_xticklabels([grp.capitalize() for grp in self.sens_grps])
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
            else:
                sns.barplot(x="positive_perc",
                        y = "a",
                        ax = ax,
                        data = plot_df,
                        color = self.sens_grps_cols[grp], 
                        order = self.sens_grps, 
                        alpha = 0.95) # To do: hvad gør den?
                ax.set_xlim((0,100))
                ax.set_yticklabels([grp.capitalize() for grp in self.sens_grps])
                ax.xaxis.set_major_formatter(mtick.FuncFormatter(abs_percentage_tick))
            error_bar(ax, plot_df, bar_mid, orientation=orientation)
            ax.set_ylabel('', fontsize = 12)
            ax.set_xlabel('', fontsize = 12)
        
        # Finishing up 
        legend_elements = [Line2D([0], [0], color=(58/255, 58/255, 58/255),
            lw=2, label='95% CI')]
        ax.legend(handles=legend_elements, frameon = True, loc = "best")
        sns.despine(ax = ax, top = True, right = True)
        ax.tick_params(left=True, labelsize=12)
        ax.set_title(title)

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

        if verbose:
            print('Plotting tSNE. This may take a while')
        
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
            palette = self.sens_grps_cols,
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
    descriptive_table = desc.get_descriptive_table()
    desc.descriptive_table_to_tex(target_tex_name='Defaulted')
    ax = desc.plot_positive_rate(orientation = 'v', title ='Fraction of Bad Credit Scores')
    desc.plot_positive_rate(orientation = 'h', title ='Fraction of Bad Credit Scores')
   

# %%
