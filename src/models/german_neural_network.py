#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from src.models.data_modules import GermanDataModule

from src.models.general_modelling_functions import (get_n_hidden_list, Net, 
BinaryClassificationTask, print_timing, objective_function)

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger('lightning').setLevel(logging.ERROR)

n_trials = 100
max_epochs = 100
#%%
if __name__ == "__main__":
    pl.seed_everything(42)
    t0 = time.time()
    output_path = 'data/predictions/german_credit_nn_pred.csv'
    param_path = 'data/predictions/german_credit_nn_pred_hyperparams.csv'

    #Save models dir 
    model_folder = 'models/german_credit/'
    params_list = []

    #Load Data Module to get dimensions for y_preds
    GM = GermanDataModule(fold = 0)
    y_pred = np.empty(GM.n_obs)
    
    print('--- Starting training ---')
    for i in range(5):
        print(f'** Fold {i} out of 5 **')

        #Loading Data Module 
        GM = GermanDataModule(fold = i)
        assert GM.fold == i

        # Find optimal model
        print('Using optuna')
        study = optuna.create_study(
            direction = 'minimize', 
            sampler = TPESampler(seed=10))
        study.optimize(
            lambda trial: objective_function(
                trial = trial, 
                dm = GM, 
                max_epochs = max_epochs), 
            n_trials = n_trials,
            show_progress_bar=False)

        # %% Train model on all data
        print('Finding best model')
        params = study.best_trial.params
        params_list.append(params)
        n_hidden_list = get_n_hidden_list(params)

        # Define network and lightning
        net = Net(
            num_features = GM.n_features, # <-- perhaps these should not be defined in data module
            num_hidden_list = n_hidden_list, 
            num_output = GM.n_output, # <-- perhaps these should not be defined in data module
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

        trainer.fit(plnet, GM)

        # Save model weigths, hparams and indexes for train and test data
        checkpoint_file = f'{model_folder}NN_german_fold_{i}'
        save_dict = {
            'model': plnet.model.eval(),
            'hparams': params,
            'fold': i}
        torch.save(save_dict, checkpoint_file)

        #test accuracy
        test_res = trainer.test(plnet, datamodule=GM)[0]
        print(f"Accuracy is: {test_res['test_acc']}")

        # Inference/predictions 
        plnet.model.eval()
        preds = plnet.model.forward(GM.test_data.X_data)
        y_pred[GM.test_idx] = preds.detach().numpy().squeeze()

    # Save data
    print('--- Done training. Saving data ---')
    output_data = (GM.raw_data[['person_id', 'credit_score', 'sex', 'age']]
        .assign(
            nn_prob = y_pred,
            nn_pred = y_pred >= 0.5
        ))
    output_data.to_csv(output_path, index = False)
    acc = accuracy_score(output_data.nn_pred, output_data.credit_score)
    params_df = pd.concat(
        [pd.DataFrame([paramdict], columns = paramdict.keys()) for paramdict in params_list])
    params_df.to_csv(param_path, index = False
    )

    print(f'Final accuracy score: {acc}')
    
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')

#%%
1+1
# %%
