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

def print_res(acc, AUROC, eval_data):
    print("---- Results ----")
    print(f"\nPredicting on {eval_data}\n")
    print(f"Accuracy = {acc}\n")
    print(f"AUROC = {AUROC}\n")
    print("------------------")

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
    print(f"Using num_workers = {num_workers}")
    
    # ---- Start: Inputs in script----
    save_metrics = True 
    save_preds = True

    model_name = "adam_dp=2e-1"
    model_type = "best" # "best" or "last"
    eval_data = "test"
    assert eval_data in ['train', 'val', 'test'], "eval_data must be 'train', 'val' or 'test'"
 
    dm = CheXpertDataModule(**{
        "target_disease":"Cardiomegaly", 
        "uncertainty_approach": "U-Zeros",
        "num_workers": num_workers, 
        "tiny_sample_data": False})

    output_path = f"data/CheXpert/predictions/{model_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    # ---- End: Inputs in Script ----

    #### Loading Checkpointed Model #######
    ckpt_folder_path = f"models/CheXpert/checkpoints_from_trainer/{model_name}/"
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
    
    print(f"model checkpoint: {model_ckpt}")
    
    pl_model = BinaryClassificationTaskCheXpert()
    pl_trained_model = pl_model.load_from_checkpoint(model_ckpt)

    ####  Predictions and Evaluation ######
    print("---- Initializing Training ----")
    trainer = pl.Trainer(
        fast_dev_run = False,
        deterministic = True,
        gpus = GPU, 
        progress_bar_refresh_rate = 0)

    # Val, Test or train data to predict on
    cols = ["patient_id"]
    if eval_data == 'val':
        df = (dm.val_data.dataset_df[cols].assign(
                y = dm.val_data.y.squeeze())
                )
        dataloader = dm.val_dataloader()
    elif eval_data == 'test':
        df = (dm.test_data.dataset_df[cols].assign(
                y = dm.test_data.y.squeeze())
                )
        dataloader = dm.test_dataloader()
    elif eval_data == 'train':
        df = (dm.train_data.dataset_df[cols].assign(
                y = dm.train_data.y.squeeze())
                )
        dataloader = dm.train_dataloader()

    print("---- Running Predictions ----")
    out_batches = trainer.predict(pl_trained_model, dataloaders = dataloader)
    scores = torch.sigmoid(torch.cat(out_batches, dim = 0))
    preds = (scores > 0.5).to(torch.int8)  

    print("---- Calculating Metrics ----")
    labels = torch.from_numpy(df.y.values).unsqueeze(dim=1).to(torch.int8)
    acc = accuracy(preds, labels)
    AUROC = auroc(preds = scores, target = labels)
    print_res(acc, AUROC, eval_data)

    if save_metrics: 
        print("---- Saving Metrics ----")
        save_dict = {"Predicted": eval_data, 
                     "Accuracy": acc.numpy(), 
                     "AUROC": AUROC.numpy()}
        (pd.DataFrame(save_dict, index = [0])
            .to_csv(f"{output_path}{eval_data}_{model_type}_metrics.csv", index=False)
        )
    if save_preds:
        print("---- Saving Predictions ----")
        (df.assign(
            y_hat = preds.numpy(), 
            scores = scores.numpy())
        .to_csv(f"{output_path}{eval_data}_{model_type}_predictions.csv", index=False)
        )

    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')
