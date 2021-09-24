#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import (get_n_hidden_list, myData, Net, BinaryClassificationTask, print_timing)

import torch
from torch.utils.data import  DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger('lightning').setLevel(logging.ERROR)

#%% Epochs and trials 
max_epochs = 50
n_trials = 500

#%% Objective function for optuna optimizer
def objective_function(trial: optuna.trial.Trial):
    logging.getLogger('lightning').setLevel(logging.ERROR)
    # Define parameters
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_hidden_list = []
    for i in range(n_layers):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(trial.suggest_int(name, 1, 20))
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.7)
    
    # Define network and lightning
    net = Net(
        num_features = n_features, 
        num_hidden_list = n_hidden_list, 
        num_output = n_output,
        p_dropout = p_dropout)
    plnet = BinaryClassificationTask(model = net, lr = lr)

    early_stopping = EarlyStopping('val_loss', patience = 3)
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, 
        max_epochs = max_epochs,
        callbacks = [early_stopping, optuna_pruning], 
        deterministic = True,
        logger = False,
        progress_bar_refresh_rate = 0, 
        gpus = GPU)

    trainer.fit(plnet, train_loader, val_loader)

    return trainer.callback_metrics['val_loss'].item()

#%%
if __name__ == "__main__":
    t0 = time.time()
    pl.seed_everything(42)

    #Load data
    file_path = 'data/processed/catalan-juvenile-recidivism/catalan-juvenile-recidivism-subset.csv'
    output_path = 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv'
    param_path = 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred_hyperparams.csv'
    raw_data = pd.read_csv(file_path, index_col=0)

    assert raw_data.isnull().sum(axis = 0).max() == 0

    #Save models dir 
    model_folder = 'models/catalan-juvenile-recidivism/'

    #Prepare data
    X = raw_data.drop(['V115_RECID2015_recid', 'id'], axis = 1)
    X = one_hot_encode_mixed_data(X)
    y = raw_data.V115_RECID2015_recid.to_numpy()
    n_features = X.shape[1]
    n_output = 1

    # Empty array for predictions and hyper parameters
    y_preds = np.empty(X.shape[0])
    y_preds[:] = np.nan
    params_list = []

    kf = KFold(n_splits=5, shuffle = True, random_state=42)

    for i, (train_val_idx, test_idx) in enumerate(kf.split(X)):
        X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size = 0.2, random_state = 42)

        # Standardize for optimization step
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Define dataloaders
        train_loader = DataLoader(
            dataset=myData(X_train, y_train), batch_size=32)
        val_loader = DataLoader(
            dataset=myData(X_val, y_val), batch_size = 32)

        # Find optimal model
        study = optuna.create_study(
        direction = 'minimize', 
        sampler = TPESampler(seed=10))
        max_minutes = 2
        study.optimize(
            objective_function, 
            #timeout = max_minutes*60,
            n_trials = n_trials,
            show_progress_bar=False)


        # %% Train model on all data
        params = study.best_trial.params
        params_list.append(params)
        n_hidden_list = get_n_hidden_list(params)

        # Define network and lightning
        net = Net(
            num_features = n_features, 
            num_hidden_list = n_hidden_list, 
            num_output = n_output,
            p_dropout = params['p_dropout'])
        plnet = BinaryClassificationTask(model = net, lr = params['lr'])

        if torch.cuda.is_available():
            GPU = 1
        else:
            GPU = None

        early_stopping = EarlyStopping('val_loss', patience = 3)
        trainer = pl.Trainer(
            fast_dev_run = False,
            log_every_n_steps = 1, 
            max_epochs = max_epochs,
            deterministic = True,
            callbacks = [early_stopping],
            progress_bar_refresh_rate = 0, 
            gpus = GPU)

        trainer.fit(plnet, train_loader, val_loader)

        # Save model weigths, hparams and indexes for train and test data
        checkpoint_file = f'{model_folder}NN_catalan_fold_{i}'
        save_dict = {
            'model': trainer.model,
            'hparams': params,
            'train_val_idx':train_val_idx, 
            'test_idx': test_idx,
            'fold': i}
        torch.save(save_dict, checkpoint_file)


        #%%
        predictions = plnet.model.forward(torch.Tensor(X_test))
        pred_binary = (predictions >= 0.5)
        acc = accuracy_score(y_test, pred_binary)
        print(f'Accuracy is: {acc}')
        y_preds[test_idx] = predictions.detach().numpy().squeeze()
# %%
    # Save data
    potential_sensitive = ['V1_sex', 'V2_nationality_type', 'V8_age',
                           'V4_area_origin', 'V6_province']
    output_cols = ['id', 'V115_RECID2015_recid'] + potential_sensitive
    output_data = (raw_data[output_cols]
            .assign(
                nn_prob = y_preds,
                nn_pred = y_preds >= 0.5
            ))
    output_data.to_csv(output_path, index = False)
    acc = accuracy_score(output_data.nn_pred, output_data.V115_RECID2015_recid)
    params_df = pd.concat(
            [pd.DataFrame([paramdict], columns = paramdict.keys()) for paramdict in params_list])
    params_df.to_csv(param_path, index = False)

    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')