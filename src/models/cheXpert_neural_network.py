#%% Imports
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.models.cheXpert_modelling_functions import(BinaryClassificationTaskCheXpert, 
    set_parameter_requires_grad)
from src.models.general_modelling_functions import print_timing
from src.models.data_modules import CheXpertDataModule

#%%
if __name__ == "__main__":
        
    ##### DEFINITIONS #######

    model_name = 'test_model'
    model_path = 'models/CheXpert/' + model_name
    
    # All defined variables below must be included into save_dict
    only_feature_extraction = True
    max_epochs = 1
    lr = 0.001
    reduce_lr_on_plateau = True
    lr_scheduler_patience = 1 
    early_stopping_patience = 3 # To do: Decide this
    
    save_dict = {
        'only_feature_extraction': only_feature_extraction,
        'max_epochs': max_epochs,
        'lr': lr,
        'reduce_lr_on_plateau': reduce_lr_on_plateau,
        'lr_scheduler_patience': lr_scheduler_patience,
        'early_stopping_patience': early_stopping_patience
        }

    model_checkpoint_callback = ModelCheckpoint(
        dirpath = f'models/CheXpert/checkpoints_from_trainer/{model_name}', 
        filename=f"{model_name}-"+"{epoch}-{step}-{val_loss:.2f}",
        save_top_k = 1, 
        save_last=True, 
        monitor='val_loss', 
        mode='min'   
    )

    ##### Setup #########
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    pl.seed_everything(42)
    t0 = time.time()

    #### Prepare data #######
    dm = CheXpertDataModule(**{"target_disease":"Cardiomegaly", "uncertainty_approach": "U-Zeros"})

    print('--- Initializing model ---') 

    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 
        'densenet121', 
        pretrained = True)

    # Setting 'reqiures_grad' to False for all layers
    set_parameter_requires_grad(
        model, feature_extracting=only_feature_extraction)

    # Changing last layer to binary classification, this will have requires_grad = True
    in_features_classifier = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features_classifier, 1)

    model = model.double() # To ensure compatibilty with dataset
    pl_model = BinaryClassificationTaskCheXpert(
        model = model, 
        lr = lr,
        feature_extract = only_feature_extraction,
        reduce_lr_on_plateau = reduce_lr_on_plateau,
        lr_scheduler_patience = lr_scheduler_patience)

    early_stopping = EarlyStopping(
        'val_loss', 
        patience = early_stopping_patience)

    print('--- Training model ---')
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, # Set this to determine when to log 
        max_epochs = max_epochs,
        deterministic = True,
        enable_checkpointing = True,
        callbacks = [model_checkpoint_callback, early_stopping],
        progress_bar_refresh_rate = 0,
        gpus = GPU)
    trainer.fit(pl_model, dm)

    print('--- Saving model and hyperparameters---')
    save_dict['model'] = pl_model.model.eval()
    torch.save(save_dict, model_path)
    
    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')


