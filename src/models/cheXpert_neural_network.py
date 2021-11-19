#%% Imports
import time
import platform
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.models.cheXpert_modelling_functions import(BinaryClassificationTaskCheXpert, 
    set_parameter_requires_grad)
from src.models.general_modelling_functions import print_timing
from src.models.data_modules import CheXpertDataModule

#%%
if __name__ == "__main__":
        
    ##### DEFINITIONS #######

    model_name = 'enzo_test'
    model_path = f'models/CheXpert/checkpoints_from_trainer/{model_name}'
    
    # All defined variables below must be included into hyper_dict
    only_feature_extraction = False
    max_epochs = 10
    lr = 0.001
    reduce_lr_on_plateau = True
    lr_scheduler_patience = 1 
    early_stopping = False
    early_stopping_patience = 3
    resume_from_checkpoint = True
    
    
    hyper_dict = {
        'only_feature_extraction': only_feature_extraction,
        'max_epochs': max_epochs,
        'lr': lr,
        'reduce_lr_on_plateau': reduce_lr_on_plateau,
        'lr_scheduler_patience': lr_scheduler_patience,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'resume_from_checkpoint': resume_from_checkpoint
        }

    model_checkpoint_callback = ModelCheckpoint(
        dirpath = model_path, 
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
    if platform.system() == 'Linux':
        n_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(n_avail_cpus-1, 8)
    else:
        num_workers = 0
    
    pl.seed_everything(42)
    t0 = time.time()

    #### Prepare data #######
    dm = CheXpertDataModule(**{
        "target_disease": "Cardiomegaly", 
        "uncertainty_approach": "U-Zeros",
        "num_workers": num_workers})

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
    logger = TensorBoardLogger(
        save_dir = 'models/CheXpert/lightning_logs', 
        name = model_name)
    logger.log_hyperparams(hyper_dict)
    if early_stopping:
        callbacks = [model_checkpoint_callback, early_stopping]
    else:
        callbacks = [model_checkpoint_callback]
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 50, # Set this to determine when to log 
        max_epochs = max_epochs,
        deterministic = True,
        enable_checkpointing = True,
        callbacks = callbacks,
        progress_bar_refresh_rate = 0,
        gpus = GPU,
        logger = logger)
    if resume_from_checkpoint:
        ckpt_path = f'{model_path}/last.ckpt'
        trainer.fit(pl_model, dm, ckpt_path = ckpt_path)
    else:
        trainer.fit(pl_model, dm)
    
    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')


