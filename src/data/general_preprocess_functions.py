# we could create a class for this - then we could have y and a defined by
# the user at init and use them in the different functions to e.g. subset X 
# from the preprocessed data 

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def one_hot_encode_mixed_data(X):
    # splitting into categorical and numerical columns 
    X_cat = X.loc[:, (X.dtypes=='object').values]
    X_num = X[X.columns.difference(X_cat.columns)]
    
    enc = OneHotEncoder()
    X_cat_one_hot_fit = enc.fit_transform(X_cat).todense()
    X_cat_one_hot = pd.DataFrame(X_cat_one_hot_fit, columns=enc.get_feature_names(X_cat.columns))
    
    # Concatenating into a final data frame 
    X_final = pd.concat([X_num, X_cat_one_hot], axis = 1)

    return X_final 
