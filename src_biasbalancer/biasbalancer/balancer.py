# %%
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import confusion_matrix, roc_curve

import biasbalancer.utils as bbutils
import biasbalancer.plots as bbplots

class BiasBalancer:
    """:class:`BiasBalancer` is a toolkit for fairness analysis of a binary classifier.  It facilitates  nuanced  fairness  analyses  taking  several  fairness criteria  into  account  enabling  the  user  to  get  a  fuller  overview  of  the  potential  interactions between fairness criteria. The fairness criteria included in :class:`BiasBalancer` are documented in the `overview table of fairness criteria`_

    :class:`BiasBalancer` consists of three levels, where each level increasingly nuances the fairness analysis. 
    
    The first level calculates a unified assessment of unfairness taking the severity of false positives relative to false negatives into account. For explanation of the computed quantity see `level_1 Documentation`_.
      
    The second level gives a comprehensive overview of disparities across sensitive groups including a barometer quantifying violations of a number of fairness criteria. See `level_2 Documentation`_.
    
    The third level includes several methods enabling further investigation into potential unfairness identified in level two. See `level_3 Documentation`_ for information about the specific analyses.

    Args:
        data (DataFrame): DataFrame containing data used for evaluation
        y_name (str): Name of target variable
        y_hat_name (str): Name of binary output variable
        r_name (str): Name of variable containing scores (predicted probabilities) 
        a_name (str): Name of sensitive variable 
        w_fp (int or float): False positive weight
        model_name (str): Name of the model or dataset used. Is used for plot titles. 

    Examples: 
        >>> from biasbalancer.balancer import BiasBalancer
        >>> from biasbalancer.get_compas_data import get_compas_data 
        >>> compas = get_compas_data(normalize_decile_scores = True)
        >>> bb  = BiasBalancer(data = compas, y_name = "two_year_recid", y_hat_name = "pred", a_name = "race", r_name = "decile_score", w_fp = 0.9)
        >>> bb.level_1()
        >>> bb.level_2()
        >>> bb.level_3(method = 'w_fp_influence')
    """

    def __init__(self, data, y_name, y_hat_name, a_name, r_name, w_fp, model_name=None, **kwargs):

        # Input checks
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f'`data` must be a dataframe. You supplied a {type(data)}')

        input_names = set([y_name, y_hat_name, a_name, r_name])
        if not input_names.issubset(set(data.columns)):
            raise ValueError(
                '`y_name`, `y_hat_name`, `a_name`, and `r_name` must be present in columns of `data`.')

        if not data[r_name].between(0, 1).all():
            raise ValueError(
                'Scores in column `r_name` must be in range [0,1]')

        y_binary = data[y_name].isin([0, 1]).all()
        y_hat_binary = data[y_hat_name].isin([0, 1]).all()
        if not (y_binary and y_hat_binary):
            raise ValueError(
                'Targets in column `y_name` and predictions in column `y_hat_name` must be binary.')

        if not isinstance(model_name, str) and model_name is not None:
            raise ValueError('`model_name` must be of type str or None.')

        self.data = data
        self.y_name = y_name
        self.y_hat_name = y_hat_name
        self.r_name = r_name
        self.a_name = a_name
        self.w_fp = w_fp
        self.model_name = model_name

        self.classifier = pd.DataFrame({
            'y': data[y_name],
            'a': data[a_name],
            'y_hat': data[y_hat_name],
            'r': data[r_name]}).reset_index(drop=True)
        self.sens_grps = np.sort(self.classifier.a.unique())
        self.n_sens_grps = len(self.sens_grps)
        self.n = data.shape[0]

        self.cm = self.get_confusion_matrix()
        self.rates = self.get_rates()
        self.WMR_rates = self.get_WMR_rates(w_fp=self.w_fp)
        self.rel_rates = self.get_relative_rates()
        self.WMR_rel_rates = self.get_relative_rates(self.WMR_rates)

        self.BBplot = bbplots.BiasBalancerPlots(self, **kwargs)

    ###############################################################
    #                  LAYER METHODS
    ###############################################################
    def level_1(self, plot=True, output_table=True):
        """First level of BiasBalancer computes *WMR* and *WMQ* for each sensitive group and presents the group with the largest *WMQ* in a visualization.
        
        .. _level_1 Documentation:

        *WMR* is the *weighted misclassification rate* and can be calculated for each sensitive group. For group :math:`a` the :math:`\\textit{WMR}_a` is computed as

        .. math::
            \\textit{WMR}_a = c(w_{fp})\\frac{w_{fp} FP_a + (1-w_{fp})FN_a}{n_a}

        where :math:`FP_a` is the number of false positives in the group, :math:`FN_a` is the number of false negatives in the group and :math:`n_a` is the total number of observations in the group. :math:`w_{fp}\in [0,1]` is the false positive weight indicating how severe a false positive is compared to a false negative. Larger values of :math:`w_{fp}` emphazises larger severity of false positives. :math:`c(w_{fp})` is a normalization constant ensuring :math:`WMR \in [0,1]`. The normalization constant is computed as 
        
        .. math:: 
                c(w_{FP}) = \min\left(\\frac{1}{w_{fp}}, \\frac{1}{1-w_{fp}}\\right).

        *WMQ* is the *weighted misclassification quotient*. Define :math:`WMR_{min} := \min_{a\in A} WMR` then *WMQ* for group :math:`a` is computed as 

        .. math:: 
            \\textit{WMQ}_a = \\frac{\\textit{WMR}_a - \\textit{WMR}_{min}}{\\textit{WMR}_{min}+\\varepsilon}\cdot100\%  \quad \\text{for $a \in A$}

        A larger *WMQ* suggests disfavoring between the groups. The small number :math:`\\varepsilon` is added to the denominator to ensure that the weighted misclassification qoutient is well-defined if :math:`\\textit{WMR}_{min}=0`.

        Args:
            plot (bool): If True, a plot is made visualizing which group has the largest *WMR*
            output_table (bool): If True, the results are returned in a dataframe
        
        Returns: 
            DataFrame: Dataframe with columns ``grp``, ``n``, ``WMR`` and ``WMQ``. The quantities are reported for each sensitive group. 
        """

        relative_wmr = self.get_relative_rates(self.WMR_rates)
        obs_counts = (bbutils.value_counts_df(self.classifier, 'a')
                      .rename(columns={'a': 'grp'}))
        l1_data = (pd.merge(relative_wmr, obs_counts)
                   .rename(columns={
                       'rate_val': 'WMR',
                       'relative_rate': 'WMQ'}))
        l1_data = l1_data[['grp', 'n', 'WMR', 'WMQ']]

        if plot:
            self.BBplot.plot_level_1(l1_data=l1_data, ax=None)

        if output_table:
            return l1_data

    def level_2(self, plot=True, output_table=True, suptitle=False):
        """Second level of BiasBalancer creates a visual overview of the unfairness in the predictions. 

        .. _level_2 Documentation:
        
        The overview consists of three elements structured in a visualization and three data frames. The three elements are: Absolute rates, relative rates and an unfairness barometer. 
        
        The absolute rates visualized are 
        
        * False Positive Rate :math:`\left(FPR=\\frac{FP}{N}\\right)`
        * False Negative Rate :math:`\left(FNR=\\frac{FN}{P}\\right)`
        * False Discovery Rate :math:`\left(FDR=\\frac{FP}{PP}\\right)`
        * False Omission Rate :math:`\left(FDR=\\frac{FN}{PN}\\right)` 

        where *FP* are false positives, *FN* are false negatives, *N* are negatives, *P* are positives, *PN* are predicted negatives and *PP* are predicted positives. 
        
        The relative rates are computed for each group and rate. The relative rates for group :math:`a` are computed as
            
        .. math::
            RR_a(r) &= \\frac{r_{a} - r_{min}}{r_{min}+\\varepsilon}\cdot100\%, \n
            r_{min} &= \min_{a\in A} r_a,

        for rates :math:`r \in \{\\textit{FPR},~\\textit{FNR},~\\textit{FDR},~\\textit{FOR},~\\textit{WMR}\}`. The small number :math:`\\varepsilon` is added to the denominator to ensure that the relative rate is well-defined if :math:`\\textit{r}_{min}=0`.

        The unfairness barometer indicates the level of unfairness present in the predictions according to different fairness criteria. The fairness criteria are summarized in the overview table below. 

        .. _overview table of fairness criteria:

        .. csv-table:: Overview table of Fairness Criteria
           :file: ../../references/overview_table.csv
           :header-rows: 1

        The quantity depicted in the unfairness barometer is called the *mean maximum relative rate (MMRR)*. Informally, this is the maximum relative rate across sensitive subgroups, if the criterion only depends on one rate. If the criterion depends on several rates it shows the mean of the different rate components instead. Formally, for fairness criterion :math:`f`, this is computed by 

        .. math::
            MMRR(f) = \\frac{1}{|R_{balanced}(f)|}\sum_{r\in R_{balanced}(f)} \max_{a \in A} RR_a(r),

        where :math:`|R_{balanced}(f)|` is the cardinality of the set of rates affecting the fairness criterion :math:`f`. This is e.g. 1 for *Equal Opportunity* and 2 for *Separation*. Which criterion depends on what rates can also be found in the overview table above.

        Args:
            plot (bool): If True, a plot is made visualizing the results
            output_table (bool): If True, the results are returned in a dataframe
            suptitle (bool): If True, the BiasBalancer.model_name is used as suptitle. Defaults to False.

        Returns: 
            tuple: Tuple containing dataframes:

                **rates** (*DataFrame*): Data frame with columns ``rate``, ``grp``, ``rate_val``, ``rate_val_lwr``, ``rate_val_upr``. The rates are FPR, FNR, FDR, FOR for each sensitive group including 95% Wilson Confidence intervals.
                
                **relative_rates** (*DataFrame*): Data frame with columns ``rate``, ``grp``, ``rate_val``, ``relative_rate``.
                
                **barometer** (*DataFrame*): Data frame with values to create the unfairness barometer. Columns are ``criterion``, ``relative_rate``, ``discriminated_grp``. 
        
        .. [HARDT2016] Hardt, M., Price, E., and Srebro, N. (2016).
           Equality of opportunity in supervised learning.
        .. [BAR2019]  Barocas, S., Hardt, M., and Narayanan, A. (2019).
           Fairness and Machine Learning.
        .. [VER2018] Verma,  S.  and  Rubin,  J.  (2018).
           Fairness  definitions  explained.
        """
        rates = (pd.concat(
            [self.rates, self.WMR_rates]
        )
            .query("rate in ['FPR', 'FNR', 'FDR', 'FOR', 'WMR']")
            .sort_values(['rate', 'grp'])
            .reset_index(drop=True))
        relative_rates = self.get_relative_rates(rates=rates)
        barometer = self.get_fairness_barometer()

        if plot:
            self.BBplot.plot_level_2(
                rates, relative_rates, barometer, suptitle=suptitle)

        if output_table:
            return rates.query("rate != 'WMR'"), relative_rates, barometer

    def level_3(self, method, plot=True, output_table=True, **kwargs):
        """Third level of BiasBalancer provides several methods enabling further investigation of the fairness analysis results from level 2. 

        .. _level_3 Documentation:

        The function includes the methods ``w_fp_influence``, ``roc_curves``, ``calibration``, ``confusion_matrix``, and ``independence_check``.

        The table below summarizes when to use each method and for what purpose: 

        +-------------------+------------------------------------+-----------------------------------------+
        |Method	            |When…                               |What                                     |
        +-------------------+------------------------------------+-----------------------------------------+
        |w_fp_influence	    |Unsure about how :math:`w_{FP}`  	 |WMQ for each group as a function of      |
        |                   |                                    |                                         |
        |                   |influences the result.              |:math:`w_{FP}`                           |
        +-------------------+------------------------------------+-----------------------------------------+
        |roc_curves         |Separation, FPR balance or equal    |The ROC-curve for each sensitive group   |
        |                   |                                    |                                         |
        |                   |opportunity is large in barometer   |                                         |
        +-------------------+------------------------------------+-----------------------------------------+
        |calibration	    |Sufficiency or predictive parity is |Calibration curve for each group.        |
        |                   |                                    |                                         |
        |                   |large in barometer                  |                                         |
        +-------------------+------------------------------------+-----------------------------------------+
        |confusion_matrix   |The dataset or a group contain      |Confusion matrix for each group          |
        |                   |                                    |                                         |
        |                   |few observations                    |                                         |
        +-------------------+------------------------------------+-----------------------------------------+
        |independence_check |Independence is large in barometer  |Fraction of predicted positives          |
        |                   |                                    |                                         |
        |                   |                                    |across groups                            |
        +-------------------+------------------------------------+-----------------------------------------+        

        Args:
            method ({'w_fp_influence', 'roc_curves', 'calibration', 'confusion_matrix', 'independence_check'}): The method to use for further analysis
            plot (bool): If True, a plot is made visualizing the results
            output_table (bool): If True, the results are returned in a dataframe
            **kwargs: Keyword arguments are passed onto the corresponding analysis and method, which is named get_`method`.

        Returns: 
            DataFrame: Data frame of which the columns depend on the method chosen. The data returned is the data used for the visualization that is created if ``plot=True``.
        """

        method_options = [
            'w_fp_influence',
            'roc_curves',
            'calibration',
            'confusion_matrix',
            'independence_check']

        if not isinstance(method, str):
            raise ValueError(
                f'`method` must be of type string. You supplied {type(method)}')

        if method not in method_options:
            raise ValueError(
                f'`method` must be one of the following: {method_options}. You supplied `method` = {method}')

        if method == 'w_fp_influence':
            w_fp_influence = self.get_w_fp_influence(plot=plot, **kwargs)
            if output_table:
                return w_fp_influence
            else:
                return None

        if method == 'roc_curves':
            roc = self.get_roc_curves(plot=plot, **kwargs)
            if output_table:
                return roc
            else:
                return None

        if method == 'calibration':
            n_bins = kwargs.get('n_bins', 5)
            calibration = self.get_calibration(plot=plot, n_bins=n_bins)
            if output_table:
                return calibration
            else:
                return None

        if method == 'confusion_matrix':
            cm = self.get_confusion_matrix(plot=plot, **kwargs)
            if output_table:
                return cm
            else:
                return None

        if method == 'independence_check':
            independence_check = self.get_independence_check(
                plot=plot, **kwargs)
            if output_table:
                return independence_check
            else:
                return None

    ###############################################################
    #                  CALCULATION METHODS
    ###############################################################

    def get_confusion_matrix(self, plot=False, **kwargs):
        """Calculate the confusion matrix pr sensitive group

        Args:
            plot (bool): If True, the confusion matrices are plotted

        Keyword Arguments:
            cm_print_n (bool): If True, the number of observations is shown in each cell. Defaults to False. 
        
        Returns: 
            DataFrame: Data frame in long format with information about false positives (FP), false negatives (FN), true positives (TP) and true negatives (TN) across the sensitive groups. The data frame includes the columns ``a``, ``type_obs``, ``number_obs``, ``fraction_obs``.
        """
        # Make dictionary of confusion matrices
        cm = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            cm_sklearn = confusion_matrix(
                y_true=df_group.y,
                y_pred=df_group.y_hat)
            cm[grp] = bbutils.cm_matrix_to_dict(cm_sklearn)

        # Convert into dataframe in long format
        cm_df = (pd.DataFrame(cm)
                 .reset_index()
                 .rename(columns={'index': 'type_obs'})
                 .melt(id_vars='type_obs', var_name='a', value_name='number_obs'))

        # Calculate marginal values
        agg_df = (self.classifier.groupby('a')
                  .agg(
            P=("y", bbutils.N_pos),
            N=("y", bbutils.N_neg),
            PP=("y_hat", bbutils.N_pos),
            PN=("y_hat", bbutils.N_neg))
            .reset_index()
            .melt(id_vars='a', var_name='type_obs', value_name='number_obs')
        )

        # Combine the two
        cm_df = pd.concat([cm_df, agg_df]).reset_index(drop=True)

        # Calculate fractions
        df = pd.DataFrame(
            columns=['a', 'type_obs', 'number_obs', 'fraction_obs'])
        for grp in self.sens_grps:
            n_grp = self.classifier.a.value_counts()[grp]
            tmp_df = (cm_df.query(f"a=='{grp}'")
                      .assign(fraction_obs=lambda x: x.number_obs/n_grp)
                      )
            df = df.append(tmp_df)

        df.reset_index(inplace=True, drop=True)

        if plot:
            cm_print_n = kwargs.get('cm_print_n', False)
            self.BBplot.plot_confusion_matrix(df, cm_print_n=cm_print_n)

        return df

    def get_rates(self, plot=False):
        """Calculate group wise rates.
        The groupwise rates computed are FPR, FNR, FDR and FOR. These are listed in `level_2 Documentation`_. 

        Args: 
            plot (bool): If True, the rates are visualized. 

        Returns:
            Dataframe:  Data frame with the calculated rates and 95% Wilson confidence intervals by group. The following rates are returned: FNR, FPR, FDR, FOR, PN/n, and PP/n. The Data frame includes the columns ``rate``, ``grp``, ``rate_val``, ``rate_val_lwr`` and ``rate_val_upr``.
        """
        rates = []
        for grp in self.sens_grps:
            TP, FN, FP, TN = bbutils.extract_cm_values(self.cm, grp)
            rates_grp = (pd.DataFrame(
                [['FNR', FN, (TP + FN)],
                 ['FPR', FP, (TN + FP)],
                 ['FDR', FP, (TP + FP)],
                 ['FOR', FN, (TN + FN)],
                 ['PN/n', (TN+FN), (TP+FP+TN+FN)],
                 ['PP/n', (TP+FP), (TP+FP+TN+FN)]],
                columns=['rate', 'numerator', 'denominator'])
                .assign(
                    grp=grp,
                    rate_val=lambda x: x.numerator/x.denominator,
                    rate_val_lwr=lambda x: bbutils.wilson_confint(
                        x.numerator, x.denominator, 'lwr'),
                    rate_val_upr=lambda x: bbutils.wilson_confint(
                        x.numerator, x.denominator, 'upr'))
            )
            rates.append(rates_grp)

        rates = (pd.concat(rates)
                 .reset_index(drop=True)
                 .drop(columns=['numerator', 'denominator']))

        if plot:
            self.BBplot.plot_rates(rates)

        return rates

    def get_relative_rates(self, rates=None, rate_names=None, plot=False):
        """Calculate relative difference in rates between group rate 
        and minimum rate.

        Args:
            rates (DataFrame): Contains data frame with rates from which relative rates should be calculated. The dataframe must contain the columns ``rate``, ``grp``, and ``rate_val``
            rate_names (list): list of names of rates to calculate relative rates of
        
        Returns: 
            DataFrame: Data frame with relative rates per group. The rates returned are FPR, FNR, FDR, and FOR. The data frame includes the columns ``rate``, ``grp``, ``rate_val`` and ``relative rate``. 
        """
        def get_minimum_rate(group):
            group['min_rate'] = group['rate_val'].agg('min')
            return group

        if rates is not None and rate_names is not None:
            raise ValueError('Supply either `rates` or `rate_names`')

        if rate_names == [np.nan]:
            return None
        elif rate_names is not None:
            rates = self.rates[self.rates.rate.isin(rate_names)]
        elif rates is None and rate_names is None:
            rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
            rates = self.rates[self.rates.rate.isin(rate_names)]

        # Calculate relative rates
        epsilon = np.finfo(float).eps
        rel_rates = (rates
                     .groupby(by='rate')
                     .apply(get_minimum_rate)
                     .assign(
                         relative_rate=lambda x:
                         (x.rate_val-x.min_rate)/(x.min_rate+epsilon)*100)
                     .loc[:, ['rate', 'grp', 'rate_val', 'relative_rate']]
                     .reset_index(drop=True))

        if plot:
            self.BBplot.plot_relative_rates(rel_rates)

        return rel_rates

    def get_WMR_rates(self, w_fp=None):
        """Calculate weighted misclassification rate (WMR) by group

        Args:
            w_fp (int or float): False positive weight. 
        
        Returns: 
            DataFrame: Data frame with columns ``grp``, ``rate`` and ``rate_val``. The rate computed is WMR per group. 
        """
        if w_fp is None:
            w_fp = self.w_fp

        WMR = pd.DataFrame({
            'grp': self.sens_grps,
            'rate': 'WMR'})
        WMR['rate_val'] = [calculate_WMR(self.cm, grp, w_fp)
                           for grp in WMR.grp]
        return WMR

    def get_fairness_barometer(self, plot=False):
        """ Calculate the fairness barometer

        For further explanation of the values and fairness criteria expressed by the fairness barometer see `level_2 Documentation`_.

        Args: 
            plot (bool): If True, the result is visualized. 
        
        Returns: 
            DataFrame: Data frame with columns ``criterion``, ``relative_rate``, ``discriminated_grp``. 
        """

        # Decide unfavorable outcome used for independence measure
        if self.w_fp > 0.5:
            independence_measure = 'PP/n'
        elif self.w_fp < 0.5:
            independence_measure = 'PN/n'
        else:
            # Independence not measured if w=0.5
            independence_measure = np.nan

        fairness_crit = pd.DataFrame([
            ['Independence', independence_measure],
            ['Separation', 'FPR'],
            ['Separation', 'FNR'],
            ['FPR balance', 'FPR'],
            ['Equal opportunity', 'FNR'],
            ['Sufficiency', 'FDR'],
            ['Sufficiency', 'FOR'],
            ['Predictive parity', 'FDR'],
            ['WMR balance', 'WMR']],
            columns=['criterion', 'rate']).dropna()

        rel_independence = self.get_relative_rates(
            rate_names=[independence_measure])
        all_data = (pd.concat([self.WMR_rel_rates, self.rel_rates, rel_independence])
                    .merge(fairness_crit))
        idx_discrim = (all_data
                       .groupby(by=['rate', 'criterion'], as_index=False)
                       .relative_rate
                       .idxmax())  # Idx for discriminated groups
        fairness_barometer = (all_data.loc[idx_discrim.relative_rate]
                              .groupby(by='criterion', as_index=False)
                              .agg({
                                  'relative_rate': 'mean',
                                  'grp': lambda x: list(pd.unique(x))})
                              .rename(columns={'grp': 'discriminated_grp'})
                              .sort_values('relative_rate', ascending=False))

        if plot:
            self.BBplot.plot_fairness_barometer(fairness_barometer)

        return fairness_barometer

    def get_roc_curves(self, plot=False, **kwargs):
        """ Calculates ROC curves by sensitive group

        Args:
            plot (bool): If True, the result is visualized.

        Keyword Arguments:
            threshold (int, float or dict): The threshold value(s) used for classification (will be marked by dots on the ROC-curves). Can be a single number in [0,1] or a dict with a threshold for each sensitive group. Defaults to 0.5. 

        Returns: 
            DataFrame: Data frame with roc curve information for each sensitive group in long format. The columns are ``fpr``, ``tpr``, ``threshold``, ``sens_grp``.
        """

        roc_list = []
        for grp in self.sens_grps:
            data_grp = self.classifier[self.classifier.a == grp]
            fpr, tpr, thresholds = roc_curve(
                y_true=data_grp.y,
                y_score=data_grp.r)
            roc_grp = (
                pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr,
                    'threshold': thresholds,
                    'sens_grp': grp}))
            roc_list.append(roc_grp)
        roc = pd.concat(roc_list).reset_index(drop=True)

        if plot:
            threshold = kwargs.get('threshold', 0.5)
            self.BBplot.plot_roc_curves(roc, threshold=threshold)
        return roc

    def get_independence_check(self, plot=False, **kwargs):
        """Get predicted positive rates and confidence intervals per 
        sensitive group

        Args:
            plot (bool): If True, the results are visualized

        Keyword Args:
            orientation ({'h', 'v'}): Orientation of plot. Defaults to horizontal ('h'). 

        Returns: 
            DataFrame: Data frame with columns ``a``, ``N``, ``N_predicted_label``, ``frac_predicted_label``, ``conf_lwr``, ``conf_upr`` and ``label``. 
        """

        # If w_fp >= 0.5, the 'positive' label is unfavourable and therefore plotted.
        if self.w_fp >= 0.5:
            df = (self.classifier
                  .groupby("a", as_index=False)
                  .agg(
                      N=("y_hat", "count"),
                      N_predicted_label=("y_hat", bbutils.N_pos),
                      frac_predicted_label=("y_hat", bbutils.frac_pos))
                  .assign(
                      conf_lwr=lambda x: bbutils.wilson_confint(
                          x.N_predicted_label, x.N, 'lwr'),
                      conf_upr=lambda x: bbutils.wilson_confint(
                          x.N_predicted_label, x.N, 'upr'),
                      label='positive'))
        else:
            df = (self.classifier
                  .groupby("a", as_index=False)
                  .agg(
                      N=("y_hat", "count"),
                      N_predicted_label=("y_hat", bbutils.N_neg),
                      frac_predicted_label=("y_hat", bbutils.frac_neg))
                  .assign(
                      conf_lwr=lambda x: bbutils.wilson_confint(
                          x.N_predicted_label, x.N, 'lwr'),
                      conf_upr=lambda x: bbutils.wilson_confint(
                          x.N_predicted_label, x.N, 'upr'),
                      label='negative'))

        if plot:
            orientation = kwargs.get('orientation', 'h')
            self.BBplot.plot_independence_check(df, orientation=orientation)

        return df

    def get_calibration(self, n_bins=5, plot=False):
        """ Calculate calibration by group

        Args:
            n_bins (int): Number of bins used to calculate the calibration curve. 
            plot (bool): If True, the result is visualized. 

        Returns: 
            DataFrame: Data frame with calibration data partitioned into the speficied number of bins. The columns are ``a``, ``bin_center``, ``y_bin_mean``, ``y_bin_se``, ``bin_size``, ``y_bin_lwr``, ``y_bin_upr``. 
        """
        # To do: Check if it fails gracefully if a bin contains zero observations
        bins = np.linspace(0, 1, num=n_bins+1)
        calibration_df = (
            self.classifier
            .assign(
                bin=lambda x: pd.cut(x.r, bins=bins),
                bin_center=lambda x: [x.bin[i].mid for i in range(x.shape[0])])
            .groupby(['a', 'bin_center'])
            .agg(
                y_bin_mean=('y', lambda x: np.mean(x)),
                y_bin_se=('y', lambda x: np.std(x)/np.sqrt(len(x))),
                bin_size=('y', lambda x: len(x)))
            .assign(
                y_bin_lwr=lambda x: x['y_bin_mean']-x['y_bin_se'],
                y_bin_upr=lambda x: x['y_bin_mean']+x['y_bin_se'])
            .reset_index()
        )

        if calibration_df.bin_size.min() < 2:
            warnings.warn(
                'One or more bins contain only one observations, and no confidence interval is shown for these bins.')

        if plot:
            self.BBplot.plot_calibration(calibration_df)
        return calibration_df

    def get_w_fp_influence(self, plot=False, **kwargs):
        """Investigate how the false positive weight influences WMQ or WMR
        
        *WMR* is the weighted misclassification rate and *WMQ* is the weighted misclassification quotient. For explanations about WMR and WMQ see `level_1 Documentation`_. 

        Args:
            plot (bool): If True, results are visualized

        Keyword Arguments: 
            plot_WMQ (bool): If True, the WMQ is plotted, and otherwise the WMR is plotted. Defaults to True. 

        Returns: 
            DataFrame: Data frame with columns ``index``, ``rate``, ``grp``, ``WMR``, ``WMQ``, and ``w_fp``. 

        """
        n_values = 100
        w_fp_values = np.linspace(0, 1, num=n_values)
        relative_wmrs = []
        for w_fp in w_fp_values:
            wmr = self.get_WMR_rates(w_fp=w_fp)
            rel_wmr = self.get_relative_rates(wmr).assign(w_fp=w_fp)
            relative_wmrs.append(rel_wmr)

        relative_wmrs = (pd.concat(relative_wmrs)
                         .reset_index()
                         .rename(columns={
                             'rate_val': 'WMR',
                             'relative_rate': 'WMQ'}))

        if plot:
            plot_WMQ = kwargs.get('plot_WMQ', True)
            self.BBplot.plot_w_fp_influence(relative_wmrs, plot_WMQ=plot_WMQ)

        return relative_wmrs

def calculate_WMR(cm, grp, w_fp):
    """Calculate the weighted misclassification rate (WMR)

    Args:
        cm (dataframe): confusion matrix as returned 
                        by :meth:`BiasBalancer.get_confusion_matrix()` in BiasBalancer
        w_fp (int or float): False positive weight. Must be in interval [0,1].

    Returns: 
        float: Weighted misclassification rate
    """

    # Input check of w_fp
    if not isinstance(w_fp, int) and not isinstance(w_fp, float):
        raise TypeError("w_fp must be a float or integer.")
    if w_fp < 0 or w_fp > 1:
        raise ValueError(
            f"w_fp must be in interval [0,1]. You supplied w_fp = {w_fp}")

    # normalization constant
    if w_fp < 0.5:
        c = 1/(1-w_fp)
    else:
        c = 1/w_fp

    TP, FN, FP, TN = bbutils.extract_cm_values(cm, grp)
    n = sum([TP, TN, FN, FP])
    wmr = c*(w_fp*FP + (1-w_fp)*FN)/n
    return wmr


# %% Main
if __name__ == "__main__":
    file_path = 'data/processed/anonymous_data.csv'
    df = pd.read_csv(file_path)
    df.head()
    
    fair_anym = BiasBalancer(
        data=df,
        y_name='y',
        y_hat_name='yhat',
        a_name='grp',
        r_name='phat',
        w_fp=0.8,
        model_name="Example Data")

    # l1 check
    l1 = fair_anym.level_1()

    # l2 check
    fair_anym.get_rates(plot=True)
    fair_anym.get_relative_rates(plot=True)
    fair_anym.get_fairness_barometer(plot=True)
    res = fair_anym.level_2(**{"suptitle": True})

    # l3 check
    w_fp_influence = fair_anym.get_w_fp_influence()
    w_fp_influence = fair_anym.get_w_fp_influence(
        plot=True, **{'relative': False})
    w_fp_influence = fair_anym.level_3(method='w_fp_influence')
    threshold_list = [{'A': 0.5, 'B': 0.6, 'C': 0.7}, 0.5]
    for threshold in threshold_list:
        roc = fair_anym.level_3(method='roc_curves', **
                                {'threshold': threshold})
        roc = fair_anym.get_roc_curves(threshold=threshold, plot=True)
    calibration = fair_anym.level_3(method='calibration', **{'n_bins': 3})
    independence = fair_anym.level_3(
        'independence_check', **{'orientation': 'h'})
    conf_mat = fair_anym.level_3(method='confusion_matrix')
    conf_mat = fair_anym.level_3(
        method='confusion_matrix', **{'cm_print_n': True})
    
# %%
