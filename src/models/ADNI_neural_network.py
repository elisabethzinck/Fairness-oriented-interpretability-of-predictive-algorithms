#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from src.models.data_modules import ADNIDataModule

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

import argparse

#%%
if __name__ == "__main__":
    ##### Setup #########
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    t0 = time.time()
    pl.seed_everything(42)

    parser = argparse.ArgumentParser(description = "ADNI NN")
    parser.add_argument("ADNI_no", help="Choose ADNI dataset 1 or 2", type = int) 
    args = parser.parse_args()

    ##### DEFINITIONS #######
    # Type of ADNI data set (1 or 2)
    ADNI_no = args.ADNI_no
    print(f"ADNI no: {ADNI_no}\n ADNI no type:{type(ADNI_no)}")
    #assert (args.ADNI_no == 1 | args.ADNI_no == 2)
    
    max_epochs = 5
    n_trials = 5
    max_layers = 3
    max_hidden = 20
    
    save_results_to_csv = True
    
    output_path = f'data/ADNI/predictions/ADNI_{ADNI_no}_nn_pred.csv'
    param_path = f'data/ADNI/predictions/ADNI_{ADNI_no}_nn_pred_hyper_params.csv'
    model_path = f'models/ADNI/NN_ADNI_{ADNI_no}'

    #### Prepare data #######
    dm = ADNIDataModule(dataset = ADNI_no)
    
    cols_to_keep = [dm.y_var] + dm.sens_vars
    output_data = (dm.processed_data['test_data'][cols_to_keep]
        .assign(
            nn_prob = np.nan,
            nn_pred = np.nan
        ))

    print('--- Finding optimal hyperparameters ---')
    study = optuna.create_study(
        direction = 'minimize', 
        sampler = TPESampler(seed=10))
    study.optimize(
        lambda trial: objective_function(
            trial, dm, max_layers, max_hidden, max_epochs), 
        n_trials = n_trials,
        show_progress_bar=False)

    print('--- Re-training best model ---')
    params = study.best_trial.params
    n_hidden_list = get_n_hidden_list(params)
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

    print('--- Testing and making predictions using best model ---')
    plnet.model.eval()
    nn_prob = (plnet.model
        .forward(dm.test_data.X_data)
        .detach().numpy().squeeze())
    assert(nn_prob.shape[0] == output_data.shape[0])
    output_data.nn_prob = nn_prob
    output_data.nn_pred = nn_prob >= 0.5

    acc = accuracy_score(output_data.nn_pred, output_data[dm.y_var])
    print(f'Final accuracy score: {acc}')

    print('--- Saving best model and predictions ---')
    save_dict = {
        'model': plnet.model.eval(),
        'hparams': params}
    torch.save(save_dict, model_path)

    if save_results_to_csv:
        output_data.to_csv(output_path, index = False)
        params_df = pd.DataFrame([params], columns = params.keys())
        params_df.to_csv(param_path, index = False)

    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')