import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.proportion import proportion_confint

##############################################
#                Functions
##############################################


def flatten_list(input_list):
    """Flattens a list of list
    
    Args: 
        input_list (list): list of lists 
    Returns: 
        list: Flattened list 
    """
    flat_list = [item for sub_list in input_list for item in sub_list]
    return flat_list


def cm_vals_to_matrix(TP, FN, FP, TN):
    """Converts values of TP, FN, FP and TN into a confusion matrix

    Creates a confusion matrix with the elements [[TP, FN], [FP, TN]]. 

    Args:
        TP (int): True Positives
        FN (int): False Negatives
        FP (int): False Positives 
        TN (int): True Negatives

    Returns: 
        array: Confusion matrix as a Numpy array of shape (2,2). 
    """
    cm_matrix = np.array(
        [
            [TP, FN],
            [FP, TN]
        ]
    )
    return cm_matrix


def cm_dict_to_matrix(cm_dict):
    """Convert a confusion dict to confusion matrix

    Args:
        cm_dict (dict): Confusion dict with keys TP, FP, TN, FN

    Returns: 
        array: Confusion matrix as a Numpy array of shape (2,2).
    """
    cm_matrix = np.array(
        [
            [cm_dict['TP'], cm_dict['FN']],
            [cm_dict['FP'], cm_dict['TN']]
        ]
    )
    return cm_matrix


def cm_matrix_to_dict(cm_matrix):
    """Convert a 2x2 confusion matrix to named dict

    Args:
        cm_matrix (array): Confusion matrix in form [[TP, FN], [FN, TN]]
    
    Returns: 
        dict: Dict with keys ``TP``, ``TN``, ``FN``, ``TN``.
    """
    TN, FP, FN, TP = cm_matrix.ravel()
    cm_dict = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    return cm_dict


def flip_dataframe(df, new_colname='index'):
    """Flips table such that first row becomes columns
    
    Args: 
        df (DataFrame): Data frame to be flipped.
        new_colname (str): Name of new column. Defaults to 'index'.
    
    Returns: 
        DataFrame: flipped data frame.
    """
    colnames = [new_colname] + df.iloc[:, 0].tolist()
    df = df.T.reset_index()
    df.columns = colnames
    df = df.iloc[1:, :]
    return df


def value_counts_df(df, col_name):
    """Helper function to use pd.value_counts() for dataframes

    Args:
        df (DataFrame): Dataframe containing column to use value_counts on
        col_names (str): Name of column to count values for. Must be present in df. 

    Returns:
        DataFrame: Data frame with column names `col_name` and `n`.
    """
    if col_name not in df.columns:
        raise ValueError(
            f'Supplied col_name {col_name} is not present in dataframe.')

    count_df = df[col_name].value_counts().to_frame().reset_index()
    count_df.columns = [col_name, 'n']

    return count_df


def label_case(snake_case):
    """Specialized helper function to replace underscore with spaces and capitalize first letter of string, but keep WMR and WMQ capitalized
    
    Args: 
        snake_case (str): String written in snake case to be reformatted. 
    
    Returns: 
        str: The reformatted string.
    """
    label_case = (snake_case
                  .replace("_", ' ')
                  .capitalize()
                  .replace('Wmr', 'WMR')
                  .replace('Wmq', 'WMQ'))
    return label_case


def wilson_confint(n_successes, n, side):
    """Calculate Wilson proportion confidence interval at an :math:`\\alpha`` level of 5%.

    Args: 
        n_successes (int): Count
        n (int): Number of observations
        side ({'lwr', 'upr'}): Should the function return lower or upper side of interval?
    
    Returns: 
        float: Upper or lower side of the confidence interval as requested by the input. 
    """
    conf = proportion_confint(
        count=n_successes,
        nobs=n,
        method="wilson")

    if side == 'lwr':
        return conf[0]
    elif side == 'upr':
        return conf[1]
    else:
        raise ValueError(
            f"`side` must be in ['lwr', 'upr']. You supplied `side` = {side}")


def extract_cm_values(df, grp):
    """Extracts TP, TN, FP and TN from long format confusion matrix

    Args: 
        df (Dataframe): confusion matrix as returned 
                        by :meth:`Bias.Balancer.get_confusion_matrix()`
        grp (str): name of sensitive group to retrieve matrix from 
    
    Returns: 
        tuple: Tuple with elements *TP* (true positives), *FN* (false negatives)*FP* (false positives), *TN* (true negatives).
    """
    cm_data = df.query(f"a=='{grp}'").set_index('type_obs')
    TP = cm_data.at['TP', 'number_obs']
    FN = cm_data.at['FN', 'number_obs']
    FP = cm_data.at['FP', 'number_obs']
    TN = cm_data.at['TN', 'number_obs']

    return TP, FN, FP, TN


def one_hot_encode_mixed_data(X):
    """Perform one hot encoding on categorical variables

    Args:
        X (dataframe): Dataframe to perform encoding on with both numerical and categorial variables

    Returns: 
        DataFrame: Data frame where all categorical variables are one hot encoded and numeric variables remain numeric
    """
    # splitting into categorical and numerical columns
    X_cat = X.loc[:, (X.dtypes == 'object').values]
    X_num = X.drop(columns=X_cat.columns)

    enc = OneHotEncoder(drop='if_binary', sparse=False)
    X_cat_one_hot_fit = enc.fit_transform(X_cat)
    X_cat_one_hot = pd.DataFrame(
        X_cat_one_hot_fit,
        columns=enc.get_feature_names(X_cat.columns),
        index=X_cat.index)

    # Concatenating into a final data frame
    X_final = pd.concat([X_num, X_cat_one_hot], axis=1)

    return X_final


################################################
#             lambda functions
################################################
def N_pos(x):
    """Helper function to use with pandas GroupBy. Calculates the number of positive elements"""
    return np.count_nonzero(x)
def N_neg(x):
    """Helper function to use with pandas GroupBy. Calculates the number of negative elements"""
    return len(x)-np.count_nonzero(x)
def frac_pos(x): 
    """Helper function to use with pandas GroupBy. Calculates the fraction of positive elements"""
    return (np.count_nonzero(x)/len(x))
def frac_neg(x): 
    """Helper function to use with pandas GroupBy. Calculates the fraction of negative elements"""
    return (len(x)-np.count_nonzero(x))/len(x)
