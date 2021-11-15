#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from src.models.data_modules import CatalanDataModule

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

#%% Epochs, trials, max_layers and max_hidden for optuna
max_epochs = 100
n_trials = 100


# Save models 
save_models_to_csv = True
save_results_to_csv = True

#%%
if __name__ == "__main__":
    # Setup 
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    t0 = time.time()
    pl.seed_everything(42)

    #Save models dir 
    output_path = 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv'
    param_path = 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred_hyperparams.csv'
    model_folder = 'models/catalan-juvenile-recidivism/'

    # load data module 
    dm = CatalanDataModule()
    # Empty array for predictions and hyper parameters
    y_pred = np.empty(dm.n_obs)
    params_list = []

    for i in range(5):
        #Load data module 
        dm.make_KFold_split(fold = i)
        assert dm.fold == i

        print('--- Finding optimal hyperparameters ---')
        study = optuna.create_study(
            direction = 'minimize', 
            sampler = TPESampler(seed=10))
        study.optimize(
            lambda trial: objective_function(
                trial, dm, max_epochs), 
            n_trials = n_trials,
            show_progress_bar=False)

        # %% Train model on all data
        print('--- Starting training ---')
        params = study.best_trial.params
        params_list.append(params)
        n_hidden_list = get_n_hidden_list(params)

        # Define network and lightning
        net = Net(
            num_features = dm.n_features, 
            num_hidden_list = n_hidden_list, 
            num_output = dm.n_output,
            p_dropout = params['p_dropout'])
        plnet = BinaryClassificationTask(model = net, lr = params['lr'])

        early_stopping = EarlyStopping('val_loss', patience = 3)
        trainer = pl.Trainer(
            fast_dev_run = False,
            log_every_n_steps = 1, 
            max_epochs = max_epochs,
            deterministic = True,
            callbacks = [early_stopping],
            progress_bar_refresh_rate = 0, 
            gpus = GPU)

        trainer.fit(plnet, dm)

        if save_models_to_csv:
            # Save model weigths, hparams and indexes for train and test data
            checkpoint_file = f'{model_folder}NN_catalan_fold_{i}'
            save_dict = {
                'model': plnet.model,
                'hparams': params,
                'fold': i}
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
    potential_sensitive = ['V1_sex', 'V8_age',
                           'V4_area_origin', 'V6_province']
    output_cols = ['id', 'V115_RECID2015_recid'] + potential_sensitive
    output_data = (dm.raw_data[output_cols]
            .assign(
                nn_prob = y_pred,
                nn_pred = y_pred >= 0.5
            ))
    acc = accuracy_score(output_data.nn_pred, output_data.V115_RECID2015_recid)
    params_df = pd.concat(
            [pd.DataFrame([paramdict], columns = paramdict.keys()) for paramdict in params_list])
    
    if save_results_to_csv:
        output_data.to_csv(output_path, index = False)
        params_df.to_csv(param_path, index = False)

    print(f'Final accuracy score: {acc}')

    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')