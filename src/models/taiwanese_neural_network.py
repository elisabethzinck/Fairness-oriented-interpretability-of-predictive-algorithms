#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

from src.models.general_modelling_functions import (get_n_hidden_list, Net,\
    BinaryClassificationTask, print_timing)
from src.models.data_modules.taiwanese_data_module import TaiwaneseDataModule

n_trials = 1
max_epochs = 10

#%% Objective function for optuna optimizer
def objective_function(trial: optuna.trial.Trial):
    
    # Define hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 2)
    n_hidden_list = []
    for i in range(n_layers):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(trial.suggest_int(name, 1, 10))
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.5)
    
    # Define network and lightning
    net = Net(
        num_features = dm.n_features, 
        num_hidden_list = n_hidden_list, 
        num_output = dm.n_output,
        p_dropout = p_dropout)
    plnet = BinaryClassificationTask(model = net, lr = lr)

    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None

    early_stopping = EarlyStopping('val_loss', patience = 3)
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

    trainer.fit(plnet, dm)

    return trainer.callback_metrics['val_loss'].item()

#%%

if __name__ == "__main__":
    pl.seed_everything(42)
    t0 = time.time()
    output_path = 'data/predictions/taiwanese_nn_pred.csv'
    param_path = 'data/predictions/taiwanese_nn_pred_hyperparams.csv'

    #Save models dir 
    model_folder = 'models/taiwanese/'

    #Load Data Module
    dm = TaiwaneseDataModule()
    y_pred = np.empty(dm.n_obs)
    
    print('--- Starting training ---')

    # Find optimal model
    print('Using optuna')
    study = optuna.create_study(
        direction = 'minimize', 
        sampler = TPESampler(seed=10))
    study.optimize(
        objective_function, 
        n_trials = n_trials,
        show_progress_bar=False)

    # %% Train model with optimal hyperparameters
    print('Finding best model')
    params = study.best_trial.params
    n_hidden_list = get_n_hidden_list(params)

    # Define network and lightning
    net = Net(
        num_features = dm.n_features, # <-- perhaps these should not be defined in data module
        num_hidden_list = n_hidden_list, 
        num_output = dm.n_output, # <-- perhaps these should not be defined in data module
        p_dropout = params['p_dropout'])
    plnet = BinaryClassificationTask(model = net, lr = params['lr'])

    # Callbacks for training
    early_stopping = EarlyStopping('val_loss', patience = 3)

    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, 
        max_epochs = max_epochs,
        deterministic = True,
        callbacks = [early_stopping],
        progress_bar_refresh_rate = 0,
        gpus = GPU)

    trainer.fit(plnet, dm)

    # Save model weigths, hparams and indexes for train and test data
    checkpoint_file = f'{model_folder}NN_taiwanese'
    save_dict = {
        'model': plnet.model.eval(),
        'hparams': params}
    torch.save(save_dict, checkpoint_file)

    #test accuracy
    test_res = trainer.test(plnet, datamodule=dm)[0]
    print(f"Accuracy is: {test_res['test_acc']}")

    # Inference/predictions 
    plnet.model.eval()
    preds = plnet.model.forward(dm.test_data.X_data)
    y_pred[dm.test_idx] = preds.detach().numpy().squeeze()

    # Save data
    print('--- Done training. Saving data ---')
    cols_to_keep = [dm.id_var, dm.y_var] + dm.sens_vars
    output_data = (dm.raw_data[cols_to_keep]
        .assign(
            nn_prob = y_pred,
            nn_pred = y_pred >= 0.5
        ))
    output_data.to_csv(output_path, index = False)
    acc = accuracy_score(output_data.nn_pred, output_data[dm.y_var])
    params_df = pd.DataFrame([params], columns = params.keys())
    params_df.to_csv(param_path, index = False)

    print(f'Final accuracy score: {acc}')
    
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')

#%%
1+1
# %%
