#%% Imports
import pandas as pd
import numpy as np

# sklearn 
from sklearn.metrics import confusion_matrix, roc_curve

# dir functions
from biasbalancer import utils
from biasbalancer.BiasBalancerPlots import BiasBalancerPlots

#%%

def calculate_WMR(cm, grp, w_fp):
    """Calculate weighted misclassification rate
    
    Args:
        cm (dataframe): long format confusion matrix as returned 
                        by get_confusion_matrix() in BiasBalancer
        w_fp (int or float): False positive error weight 
    """

    # Input check of w_fp
    if not isinstance(w_fp, int) and not isinstance(w_fp, float):
        raise TypeError("w_fp must be a float or integer.")
    if w_fp < 0 or w_fp > 1:
        raise ValueError(f"w_fp must be in [0,1]. You supplied w_fp = {w_fp}")

    # normalization constant
    if w_fp < 0.5:
        c = 1/(1-w_fp)
    else:
        c = 1/w_fp

    TP, FN, FP, TN = utils.extract_cm_values(cm, grp)
    n = sum([TP, TN, FN, FP])
    wmr = c*(w_fp*FP + (1-w_fp)*FN)/n
    return wmr

class BiasBalancer:
    def __init__(self, data, y_name, y_hat_name, a_name, r_name, w_fp, model_name = None, **kwargs):
        """Saves and calculates all necessary attributes for BiasBalancer object
        
        Args:
            data (pd.DataFrame): DataFrame containing data used for evaluation
            y_name (str): Name of target variable
            y_hat_name (str): Name of binary output variable
            r_name (str): Name of variable containing scores (must be within [0,1])
            a_name (str): Name of sensitive variable 
            w_fp (int or float): False positive error rate
            model_name (str): Name of the model or dataset used. Is used for plot titles. 
        """

        # Input checks
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'`data` must be a dataframe. You supplied a {type(data)}')
        
        input_names = set([y_name, y_hat_name, a_name, r_name])
        if not input_names.issubset(set(data.columns)): 
            raise ValueError('`y_name`, `y_hat_name`, `a_name`, and `r_name` must be present in columns of `data`.')

        if not data[r_name].between(0,1).all():
            raise ValueError('Scores in column `r_name` must be in range [0,1]')

        y_binary = data[y_name].isin([0,1]).all()
        y_hat_binary = data[y_hat_name].isin([0,1]).all()
        if not (y_binary and y_hat_binary):
            raise ValueError('Targets in column `y_name` and predictions in column `y_hat_name` must be binary.')

        # Save inputs
        self.data = data
        self.y_name = y_name
        self.y_hat_name = y_hat_name
        self.r_name = r_name
        self.a_name = a_name
        self.w_fp = w_fp
        self.model_name = model_name

        self.y = data[y_name]
        self.y_hat = data[y_hat_name]
        self.a = data[a_name]
        self.r = data[r_name]

        self.classifier = pd.DataFrame({
            'y': self.y, 
            'a': self.a, 
            'y_hat': self.y_hat, 
            'r': self.r}).reset_index(drop = True)
        self.sens_grps = np.sort(self.a.unique())
        self.n_sens_grps = len(self.sens_grps)
        self.n = data.shape[0]
        

        self.cm = self.get_confusion_matrix()
        self.rates = self.get_rates()
        self.WMR_rates = self.get_WMR_rates(w_fp = self.w_fp)
        self.rel_rates = self.get_relative_rates()
        self.WMR_rel_rates = self.get_relative_rates(self.WMR_rates)

        self.BBplot = BiasBalancerPlots(self, **kwargs)

    
    ###############################################################
    #                  LAYER METHODS
    ###############################################################
    
    def level_1(self, plot  = True, output_table = True):
        """To do: Documentation"""

        relative_wmr = self.get_relative_rates(self.WMR_rates)
        obs_counts = (utils.value_counts_df(self.classifier, 'a')
            .rename(columns = {'a': 'grp'}))
        l1_data = (pd.merge(relative_wmr, obs_counts)
            .rename(columns = {
                'rate_val': 'WMR',
                'relative_rate': 'WMQ'}))
        l1_data = l1_data[['grp', 'n', 'WMR','WMQ']]

        if plot:
            self.BBplot.plot_level_1(l1_data=l1_data, ax = None)

        if output_table:
            return l1_data
    
    def level_2(self, plot = True, output_table = True, **kwargs):
        """To do: Documentation"""
        rates = (pd.concat(
                [self.rates, self.WMR_rates]
                )
                .query("rate in ['FPR', 'FNR', 'FDR', 'FOR', 'WMR']")
                .sort_values(['rate', 'grp'])
                .reset_index(drop = True))
        relative_rates = self.get_relative_rates(rates = rates)
        barometer = self.get_fairness_barometer()
    
        if plot:
            self.BBplot.plot_level_2(rates, relative_rates, barometer, **kwargs)

        if output_table:
            return rates.query("rate != 'WMR'"), relative_rates, barometer
        
    def level_3(self, method, plot = True, output_table = True, **kwargs):
        """To do: Documentation"""

        method_options = [
            'w_fp_influence', 
            'roc_curves', 
            'calibration', 
            'confusion_matrix',
            'independence_check']

        if not isinstance(method, str):
            raise ValueError(f'`method` must be of type string. You supplied {type(method)}')
        
        if method not in method_options:
            raise ValueError(f'`method` must be one of the following: {method_options}. You supplied `method` = {method}')

        if method == 'w_fp_influence':
            w_fp_influence = self.get_w_fp_influence(plot = plot)
            if output_table:
                return w_fp_influence
            else:
                return None

        if method == 'roc_curves':
            threshold = kwargs.get('threshold', None)
            roc = self.get_roc_curves(plot = plot, threshold = threshold)
            if output_table:
                return roc
            else:
                return None

        if method == 'calibration':
            n_bins = kwargs.get('n_bins', 5)
            calibration = self.get_calibration(plot = plot, n_bins = n_bins)
            if output_table:
                return calibration
            else:
                return None

        if method == 'confusion_matrix':
            cm = self.get_confusion_matrix(plot = plot, **kwargs)
            if output_table:
                return cm
            else:
                return None
        
        if method == 'independence_check':
            independence_check = self.get_independence_check(plot = plot, **kwargs)
            if output_table:
                return independence_check
            else:
                return None


    ###############################################################
    #                  CALCULATION METHODS
    ###############################################################
    
    def get_confusion_matrix(self, plot = False, **kwargs):
        """Calculate the confusion matrix for sensitive groups"""
        # To do: Make code below more readable
        cm = {}
        for grp in self.sens_grps:
            df_group = self.classifier[self.classifier.a == grp]
            cm_sklearn = confusion_matrix(
                y_true = df_group.y, 
                y_pred = df_group.y_hat)
            cm[grp] = utils.cm_matrix_to_dict(cm_sklearn)
        
        # Making into a data frame of long format 
        data = pd.DataFrame({'a':self.a, 'y': self.y, 'y_hat':self.y_hat})
        agg_df = (data.groupby('a')
            .agg(
                P = ("y", utils.N_pos), 
                N = ("y", utils.N_neg),
                PP = ("y_hat", utils.N_pos),
                PN = ("y_hat", utils.N_neg))    
            .reset_index()     
        )
        cm_df=(utils.flip_dataframe(pd.DataFrame(cm).reset_index())
            .rename(columns={'index':'a'})
            .set_index('a')
            .join(agg_df.set_index('a'))
            .reset_index()
            .melt(id_vars = 'a',value_name='number_obs', var_name='type_obs')
        )  
        df = pd.DataFrame(columns=['a', 'type_obs', 'number_obs', 'fraction_obs'])
        for grp in self.sens_grps:
            n_grp = self.a.value_counts()[grp]
            tmp_df = (cm_df.query(f"a=='{grp}'")
                .assign(fraction_obs = lambda x: x.number_obs/n_grp)
            )
            df = df.append(tmp_df)

        df.reset_index(inplace=True, drop=True)

        if plot:
            self.BBplot.plot_confusion_matrix(df, **kwargs)

        return df

    def get_rates(self, plot = False):
        """To do: Documentation"""
        rates = []   
        for grp in self.sens_grps:
            TP, FN, FP, TN = utils.extract_cm_values(self.cm, grp)
            rates_grp = (pd.DataFrame(
                [['FNR', FN, (TP + FN)], 
                ['FPR', FP, (TN + FP)],
                ['FDR', FP, (TP + FP)],
                ['FOR', FN, (TN + FN)],
                ['PN/n', (TN+FN), (TP+FP+TN+FN)],
                ['PP/n', (TP+FP), (TP+FP+TN+FN)]], 
                columns = ['rate', 'numerator', 'denominator'])
                .assign(
                    grp = grp,
                    rate_val = lambda x: x.numerator/x.denominator,
                    rate_val_lwr = lambda x: utils.wilson_confint(
                        x.numerator, x.denominator, 'lwr'),
                    rate_val_upr = lambda x: utils.wilson_confint(
                        x.numerator, x.denominator, 'upr'))
                )
            rates.append(rates_grp)

        rates = (pd.concat(rates)
            .reset_index(drop = True)
            .drop(columns = ['numerator', 'denominator']))

        if plot:
            self.BBplot.plot_rates(rates)
        
        return rates

    def get_relative_rates(self, rates = None, rate_names = None, plot = False):
        """Calculate relative difference in rates between group rate 
        and minimum rate.

        Args:
            rates (DataFrame): Contains data frame with rates from which relative rates should be calculated
            rate_names (list): list of names of rates for which to calculate relative rates
        """
        def get_minimum_rate(group):
            group['min_rate'] = group['rate_val'].agg('min')
            return group
        
        if rate_names == [np.nan]:
            return None
        elif rate_names is not None:
            rates = self.rates[self.rates.rate.isin(rate_names)]
        elif rates is None and rate_names is None:
            rate_names = ['FPR', 'FNR', 'FDR', 'FOR']
            rates = self.rates[self.rates.rate.isin(rate_names)]
    

        # Calculate relative rates
        rel_rates = (rates
            .groupby(by = 'rate')
            .apply(get_minimum_rate)
            .assign(
                relative_rate = lambda x: 
                    (x.rate_val-x.min_rate)/x.min_rate*100)
            .loc[:, ['rate', 'grp', 'rate_val', 'relative_rate']]
            .reset_index(drop = True))

        if plot:
            self.BBplot.plot_relative_rates(rel_rates)

        return rel_rates
    
    def get_WMR_rates(self, w_fp = None):
        if w_fp is None:
            w_fp = self.w_fp

        WMR = pd.DataFrame({
            'grp': self.sens_grps,
            'rate': 'WMR'})
        WMR['rate_val'] = [calculate_WMR(self.cm, grp, w_fp) for grp in WMR.grp]
        return WMR

    def get_fairness_barometer(self, plot = False):

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
            columns = ['criterion', 'rate']).dropna()
        
        rel_independence = self.get_relative_rates(
            rate_names = [independence_measure])
        all_data = (pd.concat([self.WMR_rel_rates, self.rel_rates, rel_independence])
            .merge(fairness_crit))
        idx_discrim = (all_data
            .groupby(by = ['rate', 'criterion'], as_index = False)
            .relative_rate
            .idxmax()) # Idx for discriminated groups
        fairness_barometer = (all_data.loc[idx_discrim.relative_rate]
            .groupby(by = 'criterion', as_index = False)
            .agg({
                'relative_rate': 'mean',
                'grp': lambda x: list(pd.unique(x))})
            .rename(columns = {'grp': 'discriminated_grp'})
            .sort_values('relative_rate', ascending = False))

        if plot:
            self.BBplot.plot_fairness_barometer(fairness_barometer)

        return fairness_barometer

    def get_roc_curves(self, threshold = None, plot = False):
        # To do: Documentation
        roc_list = []
        for grp in self.sens_grps:
            data_grp = self.classifier[self.classifier.a == grp]
            fpr, tpr, thresholds = roc_curve(
                y_true = data_grp.y, 
                y_score = data_grp.r)
            roc_grp = (
                pd.DataFrame({
                    'fpr': fpr, 
                    'tpr': tpr, 
                    'threshold': thresholds,
                    'sens_grp': grp}))
            roc_list.append(roc_grp)
        roc = pd.concat(roc_list).reset_index(drop = True) 

        if plot:
            self.BBplot.plot_roc_curves(roc, threshold = threshold) 
        return roc  

    def get_independence_check(self, plot = False, **kwargs):
        """Predicted Positive rates and Confidence Intervals per 
        sensitive group"""
        
        if self.w_fp >= 0.5:
            df = (self.classifier
                .groupby("a", as_index = False)
                .agg(
                    N = ("y_hat", "count"),
                    N_predicted_label = ("y_hat", utils.N_pos),
                    frac_predicted_label = ("y_hat", utils.frac_pos))
                .assign(
                    conf_lwr = lambda x: utils.wilson_confint(
                        x.N_predicted_label, x.N, 'lwr'),
                    conf_upr = lambda x: utils.wilson_confint(
                        x.N_predicted_label, x.N, 'upr'),
                    label = 'positive'))
        else:
            df = (self.classifier
                .groupby("a", as_index = False)
                .agg(
                    N = ("y_hat", "count"),
                    N_predicted_label = ("y_hat", utils.N_neg),
                    frac_predicted_label = ("y_hat", utils.frac_neg))
                .assign(
                    conf_lwr = lambda x: utils.wilson_confint(
                        x.N_predicted_label, x.N, 'lwr'),
                    conf_upr = lambda x: utils.wilson_confint(
                        x.N_predicted_label, x.N, 'upr'),
                    label = 'negative'))

        if plot:
            orientation = kwargs.get('orientation', 'h')
            self.BBplot.plot_independence_check(df, orientation = orientation)

        return df

    def get_calibration(self, n_bins = 5, plot = False):
        """ Calculate calibration by group

        Args:
            n_bins (int): Number of bins used in calculation

        Returns:
            pd.DataFrame with calculations. To do: Should columns be described? 
        """
        # To do: Warn if no variation in one of bins (see catalan)
        bins = np.linspace(0, 1, num = n_bins+1)
        calibration_df = (
            self.classifier
            .assign(
                bin = lambda x: pd.cut(x.r, bins = bins),
                bin_center = lambda x: [x.bin[i].mid for i in range(x.shape[0])])
            .groupby(['a', 'bin_center'])
            .agg(
                y_bin_mean = ('y', lambda x: np.mean(x)),
                y_bin_se = ('y', lambda x: np.std(x)/np.sqrt(len(x))),
                bin_size = ('y', lambda x: len(x)))
            .assign(
                y_bin_lwr = lambda x: x['y_bin_mean']-x['y_bin_se'],
                y_bin_upr = lambda x: x['y_bin_mean']+x['y_bin_se'])
            .reset_index()
            )

        if plot:
            self.BBplot.plot_calibration(calibration_df)
        return calibration_df

    def get_w_fp_influence(self, plot = False, **kwargs):
        """Investigate how w_fp influences WMQ or WMR
        
        Args:
            relative (bool): Plot weighted misclassification quotient? If False, weighted misclassification rate is plotted
        """
        n_values = 100
        w_fp_values = np.linspace(0, 1, num = n_values)
        relative_wmrs = []
        for w_fp in w_fp_values:
            wmr = self.get_WMR_rates(w_fp = w_fp)
            rel_wmr = self.get_relative_rates(wmr).assign(w_fp = w_fp)
            relative_wmrs.append(rel_wmr)

        relative_wmrs = (pd.concat(relative_wmrs)
            .reset_index()
            .rename(columns = {
                'rate_val': 'WMR',
                'relative_rate': 'WMQ'}))
        
        if plot:
            self.BBplot.plot_w_fp_influence(relative_wmrs, **kwargs)
        
        return relative_wmrs
        

#%% Main
if __name__ == "__main__":
    file_path = 'data/processed/anonymous_data.csv'
    df = pd.read_csv(file_path)
    df.head()

    fair_anym = BiasBalancer(
        data = df,
        y_name = 'y',
        y_hat_name = 'yhat', 
        a_name = 'grp', 
        r_name = 'phat',
        w_fp = 0.8,
        model_name="Example Data")

    # l1 check
    l1 = fair_anym.level_1()

    # l2 check
    fair_anym.get_rates(plot = True)
    fair_anym.get_relative_rates(plot = True)
    fair_anym.get_fairness_barometer(plot = True)
    res = fair_anym.level_2(**{"suptitle":True})

    # l3 check
    w_fp_influence = fair_anym.get_w_fp_influence()
    w_fp_influence = fair_anym.get_w_fp_influence(plot = True, **{'relative': False})
    w_fp_influence = fair_anym.level_3(method = 'w_fp_influence')
    threshold_list = [{'A': 0.5, 'B': 0.6, 'C': 0.7}, 0.5]
    for threshold in threshold_list:
        roc = fair_anym.level_3(method = 'roc_curves', **{'threshold': threshold})
        roc = fair_anym.get_roc_curves(threshold = threshold, plot = True)
    calibration = fair_anym.level_3(method = 'calibration', **{'n_bins': 3})
    independence = fair_anym.level_3('independence_check', **{'orientation':'h'}) 
    conf_mat = fair_anym.level_3(method = 'confusion_matrix')
    conf_mat = fair_anym.level_3(method = 'confusion_matrix', **{'cm_print_n': True})

# %%
