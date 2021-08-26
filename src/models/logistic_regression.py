#%%
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.data.general_preprocess_functions import one_hot_encode_mixed_data

#%%

if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit.csv'
    output_path = 'data\\processed\\german_credit_pred.csv'
    data = pd.read_csv(file_path)
    
    df = data.copy()
    df['log_reg_pred'] = np.nan
    df['log_reg_prob'] = np.nan
    
    X = data[data.columns.difference(['credit_score'])]
    X = one_hot_encode_mixed_data(X)
    y = data.credit_score

    log_reg = LogisticRegression(
        solver = 'liblinear', max_iter = 1000)
    standardizer = StandardScaler()

        
    k = 10 # Number of splits
    kf = KFold(n_splits = k, random_state = 42, shuffle = True)

    for train_idx, test_idx in kf.split(X):
        X_train , X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train , y_test = y[train_idx] , y[test_idx]

        X_train = standardizer.fit_transform(X_train)
        X_test = standardizer.transform(X_test)

        log_reg.fit(X_train, y_train)
        df.loc[test_idx, 'log_reg_pred'] = log_reg.predict(X_test)
        df.loc[test_idx, 'log_reg_prob'] = log_reg.predict_proba(X_test)[:,1] 
    

    df.to_csv(output_path, index = False)

