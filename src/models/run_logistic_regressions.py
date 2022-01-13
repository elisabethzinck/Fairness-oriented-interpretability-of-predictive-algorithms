#%%
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data.general_preprocess_functions import one_hot_encode_mixed_data

from src.models.data_modules import *

#%%
def run_logistic_regression_KFold(dm):
    # Prepare for output data
    cols_to_keep = [dm.id_var] + dm.sens_vars +[dm.y_var]
    output_data = (dm.raw_data[cols_to_keep]
        .assign(
            log_reg_prob = np.nan,
            log_reg_pred = np.nan
        ))
        
    k = dm.kf.get_n_splits() # Number of splits
    for i in range(k):
        print(f'** Fold {i+1}/{k} **')
        
        # Prepare data
        dm.make_KFold_split(fold = i)
        X_train = dm.train_data.X_data
        y_train = dm.train_data.y_data
        test_idx = dm.test_idx
        X_test = dm.test_data.X_data

        # Fit model
        log_reg = LogisticRegression(
            solver = 'liblinear', 
            max_iter = 1000,
            random_state = 42)
        log_reg.fit(X_train, y_train)

        # Save predictions
        y_pred = log_reg.predict(X_test)
        y_prob = log_reg.predict_proba(X_test)[:,1] 
        output_data.loc[test_idx, 'log_reg_pred'] = y_pred
        output_data.loc[test_idx, 'log_reg_prob'] = y_prob

    # Print accuracy
    acc = accuracy_score(output_data.log_reg_pred, output_data[dm.y_var])
    print(f'Final accuracy score: {acc}')
    return output_data

def run_logistic_regression(dm):
    # Prepare for output data
    cols_to_keep = [dm.id_var] + dm.sens_vars +[dm.y_var]
    # Remaining datasets
    output_data = (dm.raw_data[cols_to_keep]
        .iloc[dm.test_idx])
    
    # Prepare data
    X_train = dm.train_data.X_data
    y_train = dm.train_data.y_data
    test_idx = dm.test_idx
    X_test = dm.test_data.X_data

    # Fit model
    log_reg = LogisticRegression(
        solver = 'liblinear', 
        max_iter = 1000,
        random_state = 42)
    log_reg.fit(X_train, y_train)

    # Save predictions
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:,1] 
    output_data['log_reg_pred'] = y_pred
    output_data['log_reg_prob'] = y_prob

    # Print accuracy
    acc = accuracy_score(output_data.log_reg_pred, output_data[dm.y_var])
    print(f'Final accuracy score: {acc}')
    return output_data

#%%
if __name__ == "__main__":
    run_german = False
    run_catalan = False
    run_taiwanese = False

    if run_german:
        print('Running German Credit Data')
        dm = GermanDataModule()
        preds = run_logistic_regression_KFold(dm)
        output_path = 'data/predictions/german_credit_log_reg.csv'
        preds.to_csv(output_path, index = False)

    if run_catalan:
        print('Running Catalan Data')
        dm = CatalanDataModule()
        preds = run_logistic_regression_KFold(dm)
        output_path = 'data/predictions/catalan_log_reg.csv'
        preds.to_csv(output_path, index = False) 

    if run_taiwanese:
        print('Running Taiwanese Data')
        dm = TaiwaneseDataModule()
        preds = run_logistic_regression(dm)
        output_path = 'data/predictions/taiwanese_log_reg.csv'
        preds.to_csv(output_path, index = False) 


# %%
