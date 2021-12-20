import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.proportion import proportion_confint

##############################################
#                Functions 
##############################################
def flatten_list(t):
    """Flattens a list of list"""
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

def cm_vals_to_matrix(TP, FN, FP, TN):
    """Converts values of TP, FN, FP and TN into a confusion matrix
    
    Args:
        TP (int): True Positives
        FN (int): False Negatives
        FP (int): False Positives 
        TN (int): True Negatives

    Returns: 
        Numpy array of shape (2,2) 
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
        Numpy array of shape (2,2) 
    """
    cm_matrix = np.array(
        [
            [cm_dict['TP'], cm_dict['FN']], 
            [cm_dict['FP'], cm_dict['TN']]
        ]
    )
    return cm_matrix

def cm_matrix_to_dict(cm_matrix):
    """
    
    Convert a 2x2 confusion matrix to named dict

    Confusion matrix must be of form [[TP, FN], [FN, TN]]
    """
    TN, FP, FN, TP = cm_matrix.ravel()
    cm_dict = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    return cm_dict


def round_func(x, base = 5):
    """Costum round function, which rounds to the nearest base. Default base = 5"""
    return(round(x/base)*base)

def flip_dataframe(df, new_colname = 'index'):
    """Flips table such that first row becomes columns"""
    colnames = [new_colname] + df.iloc[:,0].tolist()
    df = df.T.reset_index()
    df.columns = colnames
    df = df.iloc[1:, :]
    return df

def value_counts_df(df, col_name):
    """pd.value_counts() for dataframes
    
    Args:
        df (pd.DataFrame): Dataframe containing column to use value_counts on
        col_names (str): Name of column to count values for. Must be present in df. 

    Returns:
        Dataframe with column names `col_name` and `n`.
    
    """
    if col_name not in df.columns:
        raise ValueError(f'Supplied col_name {col_name} is not present in dataframe.')

    count_df = df[col_name].value_counts().to_frame().reset_index()
    count_df.columns = [col_name, 'n']

    return count_df

def label_case(snake_case):
    "Replace underscore with spaces and capitalize first letter of string, but keep WMR and WMQ capitalized"
    label_case = (snake_case
        .replace("_", ' ')
        .capitalize()
        .replace('Wmr', 'WMR')
        .replace('Wmq', 'WMQ'))
    return label_case

def wilson_confint(n_successes, n, side):
    """Calculate Wilson proportion CI
    
    Args: 
        n_successes (int): Count
        n (int): Number of observations
        side ({'lwr', 'upr'}): Return lower or upper side of interval
    """
    conf = proportion_confint(
        count = n_successes,
        nobs = n,
        method = "wilson")
    
    if side == 'lwr':
        return conf[0]
    elif side == 'upr':
        return conf[1]
    else:
        raise ValueError(f"`side` must be in ['lwr', 'upr']. You supplied `side` = {side}")
        return None

def extract_cm_values(df, grp):
    """Extracts TP, TN, FP and TN from long format confusion matrix
    
    Args: 
        df(Dataframe): long format confusion matrix as returned 
                        by get_confusion_matrix() in FairKit
        grp(str): sensitive group name 
    """
    cm_data = df.query(f"a=='{grp}' & type_obs in ['TP', 'FN', 'FP', 'TN']")
    TP, FN, FP, TN = cm_data.number_obs.to_list()

    return TP, FN, FP, TN

def one_hot_encode_mixed_data(X):
    """ To do: Documentation """
    # splitting into categorical and numerical columns 
    X_cat = X.loc[:, (X.dtypes=='object').values]
    X_num = X.drop(columns = X_cat.columns)
    
    enc = OneHotEncoder(drop='if_binary', sparse = False)
    X_cat_one_hot_fit = enc.fit_transform(X_cat)
    X_cat_one_hot = pd.DataFrame(
        X_cat_one_hot_fit, 
        columns=enc.get_feature_names(X_cat.columns), 
        index = X_cat.index)
        
    # Concatenating into a final data frame 
    X_final = pd.concat([X_num, X_cat_one_hot], axis = 1)

    return X_final 

################################################
#             lambda functions
################################################
N_pos = lambda x: np.count_nonzero(x)
N_neg = lambda x: len(x)-np.count_nonzero(x)
frac_pos = lambda x: (np.count_nonzero(x)/len(x))
frac_neg = lambda x: (len(x)-np.count_nonzero(x))/len(x)