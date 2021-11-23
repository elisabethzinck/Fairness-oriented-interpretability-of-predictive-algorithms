import torch
import os
import platform
import numpy as np
import pandas as pd
import re
import time
import pytorch_lightning as pl

from torchmetrics.functional import accuracy, auroc
from src.models.data_modules import CheXpertDataModule
from src.models.cheXpert_modelling_functions import BinaryClassificationTaskCheXpert
from src.models.data_modules import CheXpertDataModule
from src.models.general_modelling_functions import print_timing

def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False

#### Setup #######
if __name__ == '__main__':
    t0 = time.time()
    
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    if platform.system() == 'Linux':
        n_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(n_avail_cpus-1, 8)
    else:
        num_workers = 0
    
    # ---- Start: Inputs in script----
    save_metrics = True 
    save_preds = True

    model_name = "test_model"
    ckpt_folder_path = f"models/CheXpert/checkpoints_from_trainer/{model_name}/"
    model_type = "last" # "best" or "last"
    eval_data = "val"
    assert eval_data in ['train', 'val', 'test'], "eval_data must be 'train', 'val' or 'test'"
 
    dm = CheXpertDataModule(**{
        "target_disease":"Cardiomegaly", 
        "uncertainty_approach": "U-Zeros",
        "num_workers": num_workers, 
        "tiny_sample_data": True})

    output_path = f"data/CheXpert/predictions/{model_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    # ---- End: Inputs in Script ----

    #### Loading Checkpointed Model #######
    if model_type == 'best':
        files = next(os.walk(ckpt_folder_path))[2]
        best_ckpts = [f for f in files if "val_loss" in f]
        loss_list = []
        for ckpt_file in best_ckpts: 
            loss = re.findall("val_loss=(.+).ckpt", ckpt_file)[0]
            loss_list.append(float(loss))
        best_model = best_ckpts[np.argmin(loss_list)]
        model_ckpt = f"{ckpt_folder_path}{best_model}"
    elif model_type == 'last':
        model_ckpt = f"models/CheXpert/checkpoints_from_trainer/{model_name}/{model_type}.ckpt"
    else:
        raise ValueError("model_type must be 'best' or 'last'")
    
    pl_model = BinaryClassificationTaskCheXpert()
    pl_trained_model = pl_model.load_from_checkpoint(model_ckpt)

    ####  Predictions and Evaluation ######
    print("Initializing trainer")
    trainer = pl.Trainer(
        fast_dev_run = False,
        deterministic = True,
        gpus = GPU)

    # Val, Test or train data to predict on
    if eval_data == 'val':
        df = dm.val_data.dataset_df[["patient_id", "y"]]
        dataloader = dm.val_dataloader()
    elif eval_data == 'test':
        df = dm.test_data.dataset_df[["patient_id", "y"]]
        dataloader = dm.test_dataloader()
    elif eval_data == 'train':
        df = dm.train_data.dataset_df[["patient_id", "y"]]
        dataloader = dm.train_dataloader()

    print("Running Prediction")
    out_batches = trainer.predict(pl_trained_model, dataloaders = dataloader)
    print(f"output from prediction:{out_batches}")

    scores = torch.cat(out_batches, dim = 0)

    print(f"Scores: {scores}")

    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')
