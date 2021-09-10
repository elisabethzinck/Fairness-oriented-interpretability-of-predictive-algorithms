import numpy as np
def flatten_list(t):
    """Flattens a list of list"""
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

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
    TP, FN, FP, TN = cm_matrix.ravel()
    cm_dict = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    return cm_dict

def abs_percentage_tick(x, pos):
    """Return absolute percentage value w/ % as string"""
    return str(round(abs(x))) + '%'

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


