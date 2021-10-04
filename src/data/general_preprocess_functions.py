# we could create a class for this - then we could have y and a defined by
# the user at init and use them in the different functions to e.g. subset X 
# from the preprocessed data 

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def one_hot_encode_mixed_data(X):
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


def map_several_columns(data, map_dict):
    """Helper function for mapping several columns
    
    Args:
        data (pd dataframe): Dataframe with data to be mapped
        map_dict (dict of dictionaries): Dictionary containing dictionaries used for mapping. 
        
        Keys of dictionary must correspond to the names of the columns to be mapped in the dataframe. See cleaning_taiwanese_credit.py for example.
        
    """
    columns_to_be_mapped = map_dict.keys()

    for colname in columns_to_be_mapped:
        null_before = data[colname].isnull().sum()
        print(f'Mapping {colname}')
        column_map = map_dict[colname]
        data[colname] = data[colname].map(column_map)
        null_after = data[colname].isnull().sum()
        if null_before != null_after:
            raise ValueError(f'Nulls where introduced when mapping {colname}')

    return data
