#%% Imports
import time
import platform
import os

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.models.cheXpert_modelling_functions import(BinaryClassificationTaskCheXpert, 
    set_parameter_requires_grad)
from src.models.general_modelling_functions import print_timing
from src.models.data_modules import CheXpertDataModule

#%%
if __name__ == "__main__":
        
    ##### DEFINITIONS #######
    fast_dev_run = False
    tiny_sample_data = False
    
    # All defined variables below must be included into hyper_dict
    only_feature_extraction = False
    max_epochs = 20
    lr = 0.001
    reduce_lr_on_plateau = True
    lr_scheduler_patience = 1 
    early_stopping = False
    early_stopping_patience = 3
    resume_from_checkpoint = False
    optimizer = 'Adam' # Options: 'Adam' and 'SGD'
    multi_label = False

    # Passed Model_name, weight_decay and dropout with argparser 
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("model_name", help="Choose model_name", type = str)
    parser.add_argument("weight_decay", help="Choose weight decay", type = float) 
    parser.add_argument("dropout", help="Choose dropout", type = float)  
    parser.add_argument("do_ext_img_aug", help="Should there be extended im aug?", type = int) 
    parser.add_argument("do_simple_img_aug", help="Should there be simple im aug?", type = int)  
    args = parser.parse_args()    

    print(f"model_name: {args.model_name}")
    print(f"wd: {args.weight_decay}")
    print(f"dropout: {args.dropout}")
    print(f"Extended Image Augmentation: {bool(args.do_ext_img_aug)}")
    print(f"Simple Image Augmentation: {bool(args.do_simple_img_aug)}")

    model_name = args.model_name
    model_path = f'models/CheXpert/checkpoints_from_trainer/{model_name}'

    hyper_dict = {
        'only_feature_extraction': only_feature_extraction,
        'max_epochs': max_epochs,
        'lr': lr,
        'reduce_lr_on_plateau': reduce_lr_on_plateau,
        'lr_scheduler_patience': lr_scheduler_patience,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'resume_from_checkpoint': resume_from_checkpoint,
        'optimizer': optimizer,
        'multi_label': multi_label,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'do_ext_img_aug': args.do_ext_img_aug,
        'do_simple_img_aug': args.do_simple_img_aug
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

    print('--- Initializing model and datamodule ---') 

    if multi_label:
        target_disease = None
    else:
        target_disease = 'Cardiomegaly'

    dm = CheXpertDataModule(**{
        "target_disease": target_disease, 
        'multi_label': multi_label,
        "uncertainty_approach": "U-Zeros",
        "num_workers": num_workers,
        'tiny_sample_data': tiny_sample_data, 
        'extended_image_augmentation':bool(args.do_ext_img_aug),
        'simple_image_augmentation':bool(args.do_simple_img_aug)})

    pl_model = BinaryClassificationTaskCheXpert(
        lr = lr,
        feature_extract = only_feature_extraction,
        reduce_lr_on_plateau = reduce_lr_on_plateau,
        lr_scheduler_patience = lr_scheduler_patience,
        optimizer = optimizer,
        num_classes = dm.train_data.num_classes, 
        weight_decay = args.weight_decay, 
        dropout = args.dropout)


    print('--- Setup training ---')
    logger = TensorBoardLogger(
        save_dir = 'models/CheXpert/lightning_logs', 
        name = model_name)
    logger.log_hyperparams(hyper_dict)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [model_checkpoint_callback, lr_monitor_callback]
    if early_stopping:
        early_stopping = EarlyStopping(
            'val_loss', 
            patience = early_stopping_patience)
        callbacks.append(early_stopping)

    print('--- Training model ---')
    trainer = pl.Trainer(
        fast_dev_run = fast_dev_run,
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


