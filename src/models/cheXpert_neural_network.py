#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.models.cheXpert_modelling_functions import (BinaryClassificationTaskCheXpert, 
set_parameter_requires_grad)
from src.models.general_modelling_functions import print_timing
from src.models.data_modules import CheXpertDataModule

#%%
if __name__ == "__main__":
    ##### Setup #########
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    pl.seed_everything(42)
    t0 = time.time()
    
    ##### DEFINITIONS #######
    output_path = 'data/CheXpert/predictions/cheXpert_nn_pred.csv'
    param_path = 'data/CheXpert/predictions/cheXpert_nn_pred_hyperparams.csv'
    model_path = 'models/CheXpert/NN_cheXpert'

    n_trials = 100
    max_epochs = 50

    #### Prepare data #######
    dm = CheXpertDataModule()

    cols_to_keep = ["patient_id", "y"] 
    output_data = (dm.test_data.dataset_df[cols_to_keep]
        .assign(
            nn_prob = np.nan,
            nn_pred = np.nan
        ))
    
    print('--- Initializing model ---')
    lr = 0.0001  
    early_stopping = EarlyStopping('val_loss', patience = 3)

    # loading pretrained DenseNet121
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained = True)

    # Setting 'reqiures_grad' to False for all layers
    set_parameter_requires_grad(model, feature_extracting=True)

    # Changing last layer to binary classification, this will have requires_grad = True
    model.classifier = torch.nn.Linear(1024, 1)

    model = model.double()
    pl_model = BinaryClassificationTaskCheXpert(model = model, lr = lr)

    print('--- Training model ---')
    trainer = pl.Trainer(
        fast_dev_run = True,
        log_every_n_steps = 1, 
        max_epochs = max_epochs,
        deterministic = True,
        callbacks = [early_stopping],
        progress_bar_refresh_rate = 0,
        gpus = GPU)
    trainer.fit(pl_model, dm)

    print('--- Testing and making predictions using best model ---')
    pl_model.model.eval()
    batch_start_idx = 0
    for batch in dm.test_dataloader():
        print(f"shape:{batch[0].shape}")
        nn_prob = (torch.sigmoid(pl_model.model
            .forward(batch[0]))
            .detach().numpy().squeeze())
        batch_end_idx = batch_start_idx + batch[0].shape[0]
        output_data.nn_prob.iloc[batch_start_idx:batch_end_idx] = nn_prob
        batch_start_idx = batch_end_idx

    output_data = output_data.assign(nn_pred = lambda x: x.nn_prob >= 0.5)
        
    acc = accuracy_score(output_data.nn_pred, output_data['y'])
    print(f'Final accuracy score: {acc}')

    print('--- Saving best model and predictions---')
    save_dict = {'model': pl_model.model.eval()}
    torch.save(save_dict, model_path)
    output_data.to_csv(output_path, index = False)
    
    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')

#%%
1+1
# %%
