# %% Plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import transforms
import matplotlib.patches as mpatches

import colorsys
import warnings
import numpy as np
import pandas as pd

import seaborn as sns

from math import ceil, floor

import biasbalancer.utils as bbutils
# %%


class BiasBalancerPlots():
    """Plotting class used for plotting results from BiasBalancer

    Args: 
        BiasBalancer (biasbalancer.balancer.BiasBalancer): Instance from the BiasBalancer class. 

    Keyword arguments: 
        specific_col_idx (list): list of max length 11 to get specific colors form the color scheme. The items can be integers from 0-11.
    """

    def __init__(self, BiasBalancer, **kwargs):

        self.BiasBalancer = BiasBalancer

        self.w_fp = BiasBalancer.w_fp
        self.sens_grps = BiasBalancer.sens_grps
        self.n_sens_grps = BiasBalancer.n_sens_grps
        self.model_name = BiasBalancer.model_name

        # Get plot colors
        if kwargs.get("specific_col_idx"):
            assert len(kwargs.get("specific_col_idx")
                       ) == self.n_sens_grps, "list of indexes should have same length as n_sens_grps"
            self.sens_grps_cols = dict(
                zip(self.sens_grps, custom_palette(
                    specific_col_idx=kwargs.get("specific_col_idx")))
            )
        else:
            self.sens_grps_cols = dict(
                zip(self.sens_grps, custom_palette(n_colors=self.n_sens_grps))
            )

        self._tick_size = 12
        self._label_size = 13
        self._title_size = 13
        self._legend_size = 12

    #################################################
    #                  Level 1
    #################################################

    def plot_level_1(self, l1_data, ax=None):
        """ Visualize the maximum gap in WMR by text

        Args:
            l1_data (DataFrame): data frame with data returned from :meth:`biasbalancer.balancer.BiasBalancer.level_1()`
            ax (matplotlib axis): Axis to plot on. Optional. 
        """

        def format_text_level_1(ax, x, y, text_list, color_list, font_sizes, font_weights):
            """Plots a list of words with specific colors, sizes and font 
            weights on a provided axis 
            Function inspired by: https://matplotlib.org/2.0.2/examples/text_labels_and_annotations/rainbow_text.html

            Args:
                ax (matplotlib axis): axis to plot text on 
                x (float): x-coordinate of text start
                y (float): y-coordinate of text start
                text_list (list of strings): List with words to plot 
                color_list (list): List of colors for each word in text_list
                font_sizes (list): List of font sizes for each word in text_list
                font_sizes (list): List of font weigths for each word in text_list
            """
            t = ax.transData
            canvas = ax.figure.canvas

            for s, c, f, w in zip(text_list, color_list, font_sizes, font_weights):
                text = ax.text(x, y, s + " ", color=c,
                               fontsize=f, weight=w, transform=t)
                text.draw(canvas.get_renderer())
                ex = text.get_window_extent()
                t = transforms.offset_copy(
                    text._transform, x=ex.width, units='dots')
        if ax is None:
            fig = plt.figure(figsize=(6, 1.3))
            ax = fig.add_subplot(1, 1, 1)

        p_grey = desaturate((58/255, 58/255, 58/255))
        max_idx = l1_data.WMQ.idxmax()
        max_grp = l1_data.grp[max_idx]
        max_val = l1_data.WMQ[max_idx]
        max_color = desaturate(self.sens_grps_cols[max_grp])

        # Creating text lines
        line_1 = (f"The WMR of sensitive group").split()
        line_2 = (f"is {max_val:.0f}% larger than the minimum WMR").split()

        # customizing lines with group
        line_1 = line_1 + [f"'{max_grp.title()}'"]
        n_words_1 = len(line_1)
        color_list_1 = [p_grey]*n_words_1
        font_sizes_1 = [20]*n_words_1
        font_weights_1 = ['normal']*n_words_1
        color_list_1[-1] = max_color  # coloring group
        font_sizes_1[-1] = 30  # making group bigger
        font_weights_1[-1] = 'bold'

        # Costumizing lines with max_val
        n_words_2 = len(line_2)
        color_list_2 = [p_grey]*n_words_2
        font_sizes_2 = [20]*n_words_2
        font_weights_2 = ['normal']*n_words_2
        font_weights_2[1:2] = ['bold', 'bold']
        font_sizes_2[1:2] = [30, 30]

        # Plotting text on axis
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0.72, 0.85)
        ax.set_axis_off()
        sns.despine(top=True, bottom=True, left=True, right=True)
        format_text_level_1(ax, 0.02, 0.8, line_1, color_list_1,
                            font_sizes_1, font_weights_1)
        format_text_level_1(ax, 0.02, 0.74, line_2, color_list_2,
                            font_sizes_2, font_weights_2)

    #################################################
    #                  Level 2
    #################################################

    def plot_level_2(self, rates, relative_rates, barometer, suptitle=False):
        """Create level 2 plot

        Args: 
            rates (DataFrame): DataFrame as returned by :meth:`biasbalancer.balancer.BiasBalancer.get_rates()`
            relative_rates (DataFrame): DataFrame as returned by :meth:`biasbalancer.balancer.BiasBalancer.get_relative_rates()`
            barometer (DataFrame): DataFrame as returned by :meth:`biasbalancer.balancer.BiasBalancer.get_fairness_barometer()`
            suptitle (bool): If True, the :attr:`biasbalancer.balancer.BiasBalancer.model_name` is used as suptitle. Defaults to False. 

        """
        gs = GridSpec(nrows=10, ncols=3)
        f = plt.figure(figsize=(22, 6))
        ax0 = f.add_subplot(gs[:, 0])
        ax2 = f.add_subplot(gs[5:, 1:2])
        ax1 = f.add_subplot(gs[0:4, 1:2], sharex=ax2)

        self.plot_rates(rates, ax=ax0)
        self.plot_relative_rates(relative_rates, l2_plot=True, ax=ax1)
        self.plot_fairness_barometer(barometer, ax=ax2)

        f.subplots_adjust(wspace=0.5, hspace=0.7)

        if suptitle:
            f.suptitle(f"{self.model_name}",
                       x=0.35, y=0.98,
                       fontweight='bold',
                       fontsize=self._title_size+1)

    def plot_rates(self, rates, ax=None):
        """Plot FPR, FNR, FDR, FOR for each group including 95% Wilson confidence intervals. 

        Args: 
            rates (DataFrame): As returned by :meth:`biasbalancer.balancer.BiasBalancer.get_rates()`
            ax (matplotlib axis): Axis to be plotted on. Optional. 
        """

        rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
        rates = rates[rates.rate.isin(rate_names)]

        alpha_weights = get_alpha_weights(self.w_fp)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            x='rate', y='rate_val',
            hue='grp', palette=self.sens_grps_cols,
            data=rates,
            order=rate_names,
            ax=ax)
        ax.legend(loc='best', frameon=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Group rates', fontsize=14, loc='left')
        ax.set_ylim(0, 1)
        sns.despine(ax=ax, bottom=True, top=True, right=True)
        ax.tick_params(labelsize=12)

        # Set alpha values and error bars manually
        # order = from left to right by group
        containers = bbutils.flatten_list(ax.containers)
        rate_names_seq = rate_names*self.n_sens_grps
        groups = bbutils.flatten_list([[grp]*4 for grp in self.sens_grps])
        for bar, rate, grp in zip(containers, rate_names_seq, groups):
            alpha = alpha_weights[rate]
            bar.set_alpha(alpha)

            error_df = rates[(rates.grp == grp) & (rates.rate == rate)]
            ax.vlines(
                x=bar.get_x() + bar.get_width()/2,
                ymin=error_df.rate_val_lwr,
                ymax=error_df.rate_val_upr,
                colors='#6C757D',
                alpha=alpha)

        return ax

    def plot_relative_rates(self, relative_rates, l2_plot=False, ax=None):
        """Plot the rate ratio for each sensitive groups

        Args:
            relative_rates (DataFrame): DataFrame as returned by :meth:`biasbalancer.balancer.BiasBalancer.get_relative_rates()`
            ax (matplotlib axis): Axis to plot on. Optional.  
            l2_plot (bool): If True, format plot specifically for second level visualization.
        """
        # Manually sets the vertical spacing in the plot
        if l2_plot:
            rate_positions = {'WMR': 1, 'FPR': 0.8,
                              'FNR': 0.6, 'FDR': 0.4, 'FOR': 0.2}
            alpha_weights = get_alpha_weights(self.w_fp)
            ylims = (.125, 1.125)
        else:
            names = relative_rates.rate.unique()
            n_rates = len(names)
            positions = np.linspace(0.2, stop=n_rates*0.2, num=n_rates)
            rate_positions = {r: p for (r, p) in zip(names, positions)}
            alpha_weights = {r: 1 for r in names}
            ylims = (0.125, n_rates*0.2+0.125)

        plot_df = (relative_rates
                   .assign(
                       rate_position=lambda x: x.rate.map(rate_positions),
                       alpha=lambda x: x.rate.map(alpha_weights)))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.hlines(
            y='rate_position', xmin=0, xmax='relative_rate',
            data=plot_df,
            color='lightgrey', linewidth=1,
            zorder=1)
        for _, alpha in enumerate(plot_df.alpha.unique()):
            sns.scatterplot(
                data=plot_df[plot_df['alpha'] == alpha],
                x='relative_rate', y='rate_position', hue='grp',
                palette=self.sens_grps_cols,
                legend=False,
                ax=ax,
                marker='o', alpha=alpha, s=150,
                zorder=2)
        ax.set_yticks(list(rate_positions.values()))
        ax.set_yticklabels(list(rate_positions.keys()))
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('Relative rates', fontsize=14, loc='left')
        _, xmax = ax.get_xlim()
        ax.set_xlim(left=-0.05*xmax)  # To see all of leftmost dots
        ax.set_ylim(ylims)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        sns.despine(ax=ax, left=True, top=True, right=True)
        ax.tick_params(left=False, labelsize=12)

    def plot_fairness_barometer(self, fairness_barometer, ax=None):
        """Plot fairness barometer

        Args: 
            fairness_barometer (DataFrame): As returned by :meth:`biasbalancer.balancer.BiasBalancer.get_fairness_barometer()`
            ax (matplotlib axis): Axis to plot on. Optional. 
            """
        # To do: Make this more readable
        plot_df = fairness_barometer.copy(deep=True)
        plot_df['grey_bar'] = [rr if rr <=
                               20 else 20 for rr in plot_df['relative_rate']]

        if ax is None:
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(1, 1, 1)

        sns.barplot(
            x='relative_rate', y='criterion',
            data=plot_df,
            ax=ax, zorder=1)
        ax.axvline(x=20, color='grey', zorder=2, linewidth=0.5)
        sns.barplot(
            x='grey_bar', y='criterion',
            data=plot_df,
            ax=ax, zorder=2, color="#EBEBEB", edgecolor="#EBEBEB")
        add_colors_with_stripes(
            ax=ax,
            color_dict=self.sens_grps_cols,
            color_variable=plot_df.discriminated_grp)

        # Adjustments and labels
        _, xmax = ax.get_xlim()
        max_rr = plot_df.relative_rate.max()
        ax.set_xlim(left=-0.05*xmax, right=max_rr + 0.25*max_rr)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Unfairness barometer', fontsize=14, loc='left')
        sns.despine(ax=ax, left=True, top=True, right=True)
        ax.tick_params(left=False, labelsize=12)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        # Legend
        patches = get_BW_fairness_barometer_legend_patches(
            plot_df=plot_df)
        leg = ax.legend(
            handles=patches, loc='lower right',
            title='',  prop={'size': self._legend_size-3},
            bbox_to_anchor=(1.05, 0),
            frameon=True)
        leg._legend_box.align = "left"

    #################################################
    #                  Level 3
    #################################################

    def plot_confusion_matrix(self, cm, cm_print_n=False):
        """ Plots confusion matrix by sensitive group

        Args:
            cm (dataframe): Confusion matrix as returned by :meth:`biasbalancer.balancer.BiasBalancer.get_confusion_matrix()`
            cm_print_n (bool): If True, the number of observations is shown in each cell. Defaults to False. 
        """
        n_grps = len(self.sens_grps)

        # TODO: Fix grid, when we have e.g. 4 groups
        # make gridspec for groups
        ncols = min(n_grps, 3)
        nrows = ceil(n_grps/ncols)
        gs = GridSpec(nrows=nrows, ncols=ncols)
        f = plt.figure(figsize=(4*ncols, 4*nrows))

        # One plot for each sensitive group
        for i, grp in enumerate(self.sens_grps):
            TP, FN, FP, TN = bbutils.extract_cm_values(cm, grp)
            n_obs = sum([TP, FN, FP, TN])
            grp_cm = bbutils.cm_vals_to_matrix(TP, FN, FP, TN)

            N, P, PN, PP = (cm
                            .query(f'a == "{grp}" & type_obs in ["PP", "PN", "P", "N"]')
                            .sort_values(by='type_obs')
                            .fraction_obs*100
                            )

            cmap = sns.light_palette('#007EA7', as_cmap=True)

            # Specifying axis for heatmap
            row_idx = floor(i/3)
            col_idx = i % 3
            ax = f.add_subplot(gs[row_idx, col_idx])

            sns.heatmap(
                grp_cm/n_obs*100,
                annot=True,
                cmap=cmap,
                vmin=0, vmax=100,
                cbar=False,
                xticklabels=[f'Predicted\nPositive ({PP:.0f}%) ',
                             f' Predicted\n Negative ({PN:.0f}%)'],
                yticklabels=[f'  Actual Positive\n({P:.0f}%)\n',
                             f'Actual Negative  \n({N:.0f}%)\n'],
                annot_kws={'size': 15})

            # Adjust figure labels
            names = ['TP', 'FN', 'FP', 'TN']
            vals = [TP, FN, FP, TN]
            coords = [(0.25, 0.7), (1.25, 0.7), (0.25, 1.7),
                      (1.25, 1.7)]  # coords for text
            for name, val, coord, a in zip(names, vals, coords, ax.texts):
                old_text = a.get_text()
                new_text = f"{name}: {old_text}%"
                a.set_text(new_text)
                if cm_print_n:
                    ax.text(x=coord[0],
                            y=coord[1],
                            s=f"(n={val})",
                            **{"c": a.get_c(), "fontsize": 11})

            # Centering tick labels on y axis
            for label in ax.get_yticklabels():
                label.set_ha('center')
            # Titles and font size
            ax.tick_params(axis='both', labelsize=11)
            plt.ylabel(None)
            plt.xlabel(None)
            plt.title(f'{str.title(grp)} (N = {n_obs})', size=self._title_size)
            f.subplots_adjust(wspace=0.4, hspace=0.4)
            if self.model_name != None:
                f.suptitle(f"{self.model_name}",
                           fontsize=self._title_size, horizontalalignment='center',
                           y=1.05)

    def plot_w_fp_influence(self, plot_df, plot_WMQ=True):
        """ Plots the influence of w_fp on WMQ or WMR. 
        
        *WMR* is the weighted misclassification rate and *WMQ* is the weighted misclassification quotient. For explanations about WMR and WMQ see `level_1 Documentation`_. 

        Args: 
            plot_df (dataframe): As returned by :meth:`balancer.get_w_fp_influence()`
            plot_WMQ (bool): If True, WMQ is plotted, and otherwise WMR is plotted. Defaults to True. 
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if plot_WMQ:
            y = 'WMQ'
        else:
            y = 'WMR'
        sns.lineplot(
            x='w_fp', y=y, hue='grp',
            data=plot_df,
            ax=ax,
            palette=self.sens_grps_cols)
        if plot_WMQ:
            ax.axhline(y=20, color='grey', linewidth=0.5)
        sns.despine(ax=ax, top=True, right=True)
        ax.set_xlabel('$w_{fp}$', fontsize=self._label_size)
        ax.set_xlim((0, 1))
        ax.set_ylabel(bbutils.label_case(y), fontsize=self._label_size)
        ax.set_title("False Positive Weight Influence", size=self._title_size)
        ax.legend(frameon=False, prop={'size': self._legend_size})
        ax.tick_params(labelsize=self._tick_size)
        if plot_WMQ:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    def plot_roc_curves(self, roc_df, ax=None, threshold=0.5):
        """ Visualizes ROC curves by sensitive group

        Args:
            roc_df (dataframe): Dataframe as returned by :meth:`balancer.get_roc_curves()`
            ax (matplotlib axis): Ax to plot on. If None, a new axis is created. 
            threshold (int, float or dict): The threshold value(s) used for   classification (will be marked by dots on the ROC-curves). Can be a single number in [0,1] or a dict with a threshold for each sensitive group. Defaults to 0.5. 
        """

        # Make thresholds into dict
        if isinstance(threshold, int) or isinstance(threshold, float):
            t_dict = {grp: threshold for grp in self.sens_grps}
        elif isinstance(threshold, dict):
            t_dict = threshold

        # Select points corresponding to thresholds
        threshold_points = []
        for grp in self.sens_grps:
            t = t_dict[grp]
            tmp = roc_df[(t < roc_df.threshold) & (roc_df.sens_grp == grp)]
            point = tmp.loc[tmp.groupby('sens_grp').threshold.idxmin()]
            threshold_points.append(point)
        threshold_points = pd.concat(threshold_points)

        # Add CIs to points
        denominators_fpr = []
        denominators_tpr = []
        for grp in self.sens_grps:
            TP, FN, FP, TN = bbutils.extract_cm_values(
                self.BiasBalancer.cm, grp)
            denominators_fpr.append(TN+FP)
            denominators_tpr.append(TP+FN)
        threshold_points = threshold_points.assign(
            n_pos=denominators_tpr,
            n_neg=denominators_fpr,
            fpr_lwr=lambda x: bbutils.wilson_confint(
                x.fpr*x.n_neg, x.n_neg, "lwr"),
            fpr_upr=lambda x: bbutils.wilson_confint(
                x.fpr*x.n_neg, x.n_neg, "upr"),
            tpr_lwr=lambda x: bbutils.wilson_confint(
                x.tpr*x.n_pos, x.n_pos, "lwr"),
            tpr_upr=lambda x: bbutils.wilson_confint(
                x.tpr*x.n_pos, x.n_pos, "upr")
        )

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        # ROC curves
        sns.lineplot(
            x='fpr', y='tpr', hue='sens_grp',
            data=roc_df, ax=ax,
            estimator=None, alpha=0.45,
            palette=self.sens_grps_cols,
            zorder=1)
        # Threshold points
        sns.scatterplot(
            x='fpr', y='tpr',
            data=threshold_points, ax=ax,
            hue='sens_grp', s=35,
            palette=self.sens_grps_cols,
            linewidth=0,
            zorder=3, legend=False),
        # CIs
        ax.vlines(
            x=threshold_points.fpr,
            ymin=threshold_points.tpr_lwr,
            ymax=threshold_points.tpr_upr,
            colors=self.sens_grps_cols.values(), linewidth=0.5,
            zorder=2)
        ax.hlines(
            y=threshold_points.tpr,
            xmin=threshold_points.fpr_lwr,
            xmax=threshold_points.fpr_upr,
            colors=self.sens_grps_cols.values(), linewidth=0.5,
            zorder=2)

        ax.plot([0, 1], [0, 1], color='grey', linewidth=0.5)

        ax.set_xlabel('False positive rate', size=self._label_size)
        ax.set_ylabel('True positive rate', size=self._label_size)
        ax.set_title('ROC Curves (Analysis of Separation)',
                     size=self._title_size)
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(labelsize=self._tick_size)

        # Get legend
        patches = []
        for grp, col in self.sens_grps_cols.items():
            tau = round(t_dict[grp], 3)
            patch = Line2D(
                [], [],
                color=col, marker='o', markersize=10,
                markeredgecolor='white',
                label=f'{grp} ($\\tau$ = {tau})')
            patches.append(patch)
        ax.legend(
            handles=patches,
            frameon=False, loc='lower right',
            prop={'size': self._legend_size},
            markerfirst=False)

    def plot_independence_check(self, df, ax=None, orientation='h'):
        """ Bar plot of the percentage of predicted positives per
        sensitive group including a Wilson 95% Confidence interval. 

        Args:
            df (Dataframe): As returned by :meth:`biasbalancer.balancer.BiasBalancer.get_independence_check()` 
            ax (matplotlib axis): Axis to plot on. If None, a new axis is created. 
            orientation ({'h', 'v'}): Orientation of plot. Defaults to horizontal ('h'). 
        """
        assert orientation in ["v", "h"], "Choose orientation 'v' or 'h'"

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Convert into percentage
        plot_data = (df
                     .assign(perc=lambda x: x.frac_predicted_label*100,
                             conf_lwr=lambda x: x.conf_lwr*100,
                             conf_upr=lambda x: x.conf_upr*100)
                     )

        perc_label = f'Predicted {df.label[0]} (%)'

        for grp in self.sens_grps:
            plot_df = plot_data.query(f'a == "{grp}"')
            assert plot_df.shape[0] == 1
            bar_mid = plot_df.index[0]

            if orientation == 'v':
                sns.barplot(y="perc",
                            x="a",
                            ax=ax,
                            data=plot_df,
                            color=self.sens_grps_cols[grp],
                            order=self.sens_grps,
                            alpha=0.95)
                ax.set_ylim((0, 100))
                ax.set_xticklabels([grp.title() for grp in self.sens_grps])
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_ylabel(perc_label, fontsize=self._label_size)
                ax.set_xlabel(None)
            else:
                sns.barplot(x="perc",
                            y="a",
                            ax=ax,
                            data=plot_df,
                            color=self.sens_grps_cols[grp],
                            order=self.sens_grps,
                            alpha=0.95)
                ax.set_xlim((0, 100))
                ax.set_yticklabels([grp.title() for grp in self.sens_grps])
                ax.xaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xlabel(perc_label, fontsize=self._label_size)
                ax.set_ylabel(None)
            error_bar(ax, plot_df, bar_mid, orientation=orientation)

        # Finishing up
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(left=True, labelsize=self._tick_size)
        ax.set_title('Independence Check', size=self._title_size)

    def plot_calibration(self, calibration_df, ax=None):
        """Plot calibration by sensitive groups

        Args:
            calibration_df (DataFrame): As returned by :meth:`biasbalancer.balancer.BiasBalancer.get_calibration()`
            ax (matplotlib axis): Axes object to plot on. Optional. 
        """

        n_bins = calibration_df.bin_center.nunique()
        muted_colors = {k: desaturate(col)
                        for (k, col) in self.sens_grps_cols.items()}
        calibration_df = (
            calibration_df
            .assign(color=lambda x: x.a.map(muted_colors))
            .sort_values(['bin_center', 'a'])
            .reset_index())

        # Add jitter to make confidence interval visible
        jitter = list(np.linspace(-0.004, 0.004, self.n_sens_grps))*n_bins
        calibration_df['bin_center_jitter'] = calibration_df['bin_center'] + jitter

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        sns.lineplot(
            x='bin_center', y='y_bin_mean', hue='a',
            data=calibration_df, ax=ax,
            palette=self.sens_grps_cols)
        sns.scatterplot(
            x='bin_center_jitter', y='y_bin_mean', hue='a',
            data=calibration_df, ax=ax, linewidth=0,
            palette=self.sens_grps_cols, legend=False)
        ax.vlines(
            x=calibration_df.bin_center_jitter,
            ymin=calibration_df.y_bin_lwr,
            ymax=calibration_df.y_bin_upr,
            colors=calibration_df.color, linewidth=0.5)
        ax.plot([0, 1], [0, 1], color='grey', linewidth=0.5)
        ax.set_xlabel('Predicted probability', fontsize=self._label_size)
        ax.set_ylabel('True probability $\pm$ SE', fontsize=self._label_size)
        ax.set_xlim((0, 1))
        ax.set_title('Calibration (Analysis of Sufficiency)',
                     size=self._title_size)
        sns.despine(ax=ax, top=True, right=True)
        ax.legend(frameon=False,
                  loc='upper left',
                  prop={"size": self._legend_size})
        ax.tick_params(labelsize=self._tick_size)


###################################
# General plotting functions
###################################
def get_alpha_weights(w_fp):
    """Return alpha weight for determining opacity of each rate. The rates are FPR, FNR, FDR and FOR. These are also listed in `level_2 Documentation`_. 
    
    Args:   
        w_fp (int or float): False positive weight. Must be in interval [0,1].

    Returns: 
        dict: Dict with weights for each rate. Rate names are keys and weights are values. 
    """
    c = 0.2  # Factor added to make colors stronger
    if w_fp == 0.5:
        alpha_weights = {'FPR': 1, 'FNR': 1, 'FDR': 1, 'FOR': 1, 'WMR': 1}
    elif w_fp > 0.5:
        alpha_weights = {'FPR': 1, 'FNR': 1+c-w_fp,
                         'FDR': 1, 'FOR': 1+c-w_fp, 'WMR': 1}
    else:
        alpha_weights = {'FPR': c+w_fp, 'FNR': 1,
                         'FDR': c+w_fp, 'FOR': 1, 'WMR': 1}
    return alpha_weights


def custom_palette(n_colors=1, specific_col_idx=None):
    """Returns a custom palette of n_colors (max 11 colors)

        The color palette has been created using `Coolors`_ 
        and added two colors "cinnamon satin (Hex #BD7585)" and "Salmon Pink (Hex: #F193A1)"

        .. _Coolors: https://coolors.co/f94d50-f3722c-f8961e-f9844a-f9c74f-a1c680-3a9278-7ab8b6-577590-206683

        Args:
            n_color: The number of desired colors max 10. Defaults to 1.
            specific_col_idx: list of desired color indexes. Defaults to None.

        Returns: 
            list: List of RGB tuples made as a custom color palette of length `n_colors` or `len(specific_col_idx)` using seaborn.
    """
    # TODO: Perhaps change idx for 8 colors to match cheXpert
    colors = ["f94d50", "f3722c", "f8961e", "f9844a", "f9c74f", "a1c680",
              "3a9278", "7ab8b6", "577590", "206683", "F193A1", "B66879"]

    max_colors = len(colors)
    assert n_colors < max_colors, f"n_colors must be less than {max_colors}"

    if specific_col_idx is None:
        if n_colors == 1:
            col_idx = [0]
        if n_colors == 2:
            col_idx = [0, 9]
        if n_colors == 3:
            col_idx = [0, 5, 9]
        if n_colors == 4:
            col_idx = [0, 2, 6, 9]
        if n_colors == 5:
            col_idx = [0, 3, 5, 7, 9]
        if n_colors == 6:
            col_idx = [0, 4, 5, 6, 7, 9]
        if n_colors == 7:
            col_idx = [0, 2, 4, 5, 6, 7, 9]
        if n_colors == 8:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 2, 4, 5, 6, 7, 8, 9]
        if n_colors == 9:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9]
        if n_colors == 10:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        col_idx = specific_col_idx
        n_colors = len(col_idx)

    hex_colors = [f'#{colors[col_idx[i]]}' for i in range(n_colors)]

    return sns.color_palette(hex_colors)


def add_colors_with_stripes(ax, color_dict, color_variable):
    """Add colors to barplot including striped colors.

    Args:
        ax (matplotlib axis): ax of plot to color
        color_dict (dict): Dict of colors with keys as color group
        color_variable (pd.Series): Series of color groups used in bars. 
            Each item in series is a list. 
    """
    muted_colors = {k: desaturate(col) for (k, col) in color_dict.items()}

    # Set colors of bars.
    plt.rcParams['hatch.linewidth'] = 8  # Controls thickness of stripes
    for bar, var in zip(ax.containers[0], color_variable):
        if len(var) == 1:
            col = muted_colors[var[0]]
            bar.set_facecolor(col)
        elif len(var) == 2:
            col0 = muted_colors[var[0]]
            col1 = muted_colors[var[1]]
            bar.set_facecolor(col0)
            bar.set_edgecolor(col1)
            bar.set_hatch('/')
            bar.set_linewidth(0)  # No outline around bar when striped
        else:
            raise IndexError('Cannot use > 2 colors for stripes in barplot')


def error_bar(ax, plot_df, bar_mid, orientation="v"):
    """Draws error bars on ax with barplot.

    Args: 
        ax(matplotlib axis): axis with barplot 
        plot_df(DataFrame): must include columns ``conf_lwr`` and ``conf_upr``
        bar_mid(float): mid point of bar to put error bar on 
        orientation({'v', 'h'}): ``v`` or ``h`` for vertical or horizontal 
    """
    # input check
    assert orientation in ["v", "h"], "Choose orientation 'v' or 'h'"
    assert "conf_lwr" in plot_df.columns, 'column "conf_lwr" must be in data'
    assert "conf_upr" in plot_df.columns, 'column "conf_upr" must be in data'

    if orientation == 'v':
        ax.vlines(x=bar_mid,
                  ymin=plot_df.conf_lwr,
                  ymax=plot_df.conf_upr,
                  colors=(58/255, 58/255, 58/255),
                  linewidth=2)
        ax.hlines(y=[plot_df.conf_lwr, plot_df.conf_upr],
                  xmin=bar_mid-0.15,
                  xmax=bar_mid+0.15,
                  colors=(58/255, 58/255, 58/255),
                  linewidth=2)

    elif orientation == 'h':
        ax.hlines(y=bar_mid,
                  xmin=plot_df.conf_lwr,
                  xmax=plot_df.conf_upr,
                  colors=(58/255, 58/255, 58/255),
                  linewidth=2)
        ax.vlines(x=[plot_df.conf_lwr, plot_df.conf_upr],
                  ymin=bar_mid-0.15,
                  ymax=bar_mid+0.15,
                  colors=(58/255, 58/255, 58/255),
                  linewidth=2)


def desaturate(color, prop=0.75):
    """Desaturate color like in default seaborn plot

    Args: 
        prop (float): Amount to desaturate
    
    Returns: 
        tuple: RGB tuple with the desaturated color.
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    s *= prop
    new_color = colorsys.hls_to_rgb(h, l, s)
    return new_color


def get_BW_fairness_barometer_legend_patches(plot_df):
    """Create black and white patches legend in fairness barometer

    Args: 
        plot_df (DataFrame): Data frame used to create fairness barometer plot

    Returns: 
        list: List of matplotlib patches to put in legend handles 
    """
    plot_df = plot_df.query("relative_rate > 20")
    n_cols = plot_df.discriminated_grp.apply(len).unique()
    patches = []
    if 1 in n_cols:
        col = '#6C757D'
        patch_1 = mpatches.Patch(color=col,
                                 label=f'One Discriminated Group')
        # Change Label if only one discrim group
        if 2 not in n_cols:
            patch_1.set_label('Discriminated Group')
        patches.append(patch_1)
    if 2 in n_cols:
        col0 = '#6C757D'
        col1 = '#ADB5BD'
        patch_2 = (mpatches.Patch(
            label=f'Two Discriminated Groups',
            facecolor=col0,
            edgecolor=col1,
            hatch='/',
            fill=True,
            linewidth=0
        ))
        patches.append(patch_2)

    patches.append(mpatches.Patch(color='#EBEBEB', label='Unfairness <20%'))

    return patches


# %%
