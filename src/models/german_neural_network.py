#%% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import (get_n_hidden_list, myData, Net, BinaryClassificationTask)

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger('lightning').setLevel(logging.ERROR)

#%% Objective function for optuna optimizer
def objective_function(trial: optuna.trial.Trial):
    logging.getLogger('lightning').setLevel(logging.ERROR)
    # Define parameters
    n_layers = trial.suggest_int('n_layers', 1, 2)
    n_hidden_list = []
    for i in range(n_layers):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(trial.suggest_int(name, 1, 10))
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.5)
    
    # Define network and lightning
    net = Net(
        num_features = n_features, 
        num_hidden_list = n_hidden_list, 
        num_output = n_output,
        p_dropout = p_dropout)
    plnet = BinaryClassificationTask(model = net, lr = lr)

    early_stopping = EarlyStopping('val_loss', patience = 3)
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, 
        max_epochs = 50,
        callbacks = [early_stopping, optuna_pruning], 
        deterministic = True,
        logger = False,
        progress_bar_refresh_rate = 0)

    trainer.fit(plnet, train_loader, val_loader)

    return trainer.callback_metrics['val_loss'].item()


# To open tensorboard (type in command line)
# tensorboard --logdir lightning_logs
#%%

if __name__ == "__main__":
    pl.seed_everything(42)

    #Load data
    file_path = 'data\\processed\\german_credit_full.csv'
    output_path = 'data\\predictions\\german_credit_nn_pred.csv'
    param_path = 'data\\predictions\\german_credit_nn_pred_hyperparams.csv'
    raw_data = pd.read_csv(file_path)

    #Save models dir 
    folder = 'src\\models\\checkpoints\\german_credit'

    #Prepare data
    X = raw_data.drop(['credit_score', 'person_id'], axis = 1)
    X = one_hot_encode_mixed_data(X)
    y = raw_data.credit_score.to_numpy()
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
            n_trials = 5,
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

        # Callbacks for training
        early_stopping = EarlyStopping('val_loss', patience = 3)
        checkpoint_filename = f'NN_german_fold_{i}'
        checkpoint_callback = ModelCheckpoint(dirpath=folder,
                                              save_weights_only=True,
                                              filename=checkpoint_filename,
                                              auto_insert_metric_name=False, 
                                              monitor='val_loss'
                                             )
        os.path.basename(checkpoint_callback.format_checkpoint_name({}, ver = None))

        trainer = pl.Trainer(
            fast_dev_run = False,
            log_every_n_steps = 1, 
            max_epochs = 50,
            deterministic = True,
            callbacks = [early_stopping, checkpoint_callback],
            progress_bar_refresh_rate = 0)

        trainer.fit(plnet, train_loader, val_loader)
    
        #%%
        predictions = plnet.model.forward(torch.Tensor(X_test))
        pred_binary = (predictions >= 0.5)
        acc = accuracy_score(y_test, pred_binary)
        print(f'Accuracy is: {acc}')
        y_preds[test_idx] = predictions.detach().numpy().squeeze()

    # Save data
    output_data = (raw_data[['person_id', 'credit_score', 'sex', 'age']]
        .assign(
            nn_prob = y_preds,
            nn_pred = y_preds >= 0.5
        ))
    output_data.to_csv(output_path, index = False)
    acc = accuracy_score(output_data.nn_pred, output_data.credit_score)
    params_df = pd.concat(
        [pd.DataFrame([paramdict], columns = paramdict.keys()) for paramdict in params_list])
    params_df.to_csv(param_path, index = False
    )

    print(f'Final accuracy score: {acc}')

#%%
1+1