import torch
import os
import platform
import numpy as np
import pandas as pd
import re
import time

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
        "num_workers": num_workers})

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
    
    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 
        'densenet121', 
        pretrained = True)
    in_features_classifier = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features_classifier, 1)
    pl_model = BinaryClassificationTaskCheXpert(model = model)
    remove_grad(pl_model)
    #print(f"Bias of ImageNet DenseNet:\n{pl_model.model.classifier.bias}")
    pl_trained_model = pl_model.load_from_checkpoint(model_ckpt)
    remove_grad(pl_trained_model)
    #print(f"Bias of {model_name}:\n{pl_trained_model.model.classifier.bias}")

    ####  Predictions and Evaluation ######
    cols = ["patient_id", "y"]
    if eval_data == "train":
        df = dm.train_data.dataset_df[cols]
        dataloader = dm.train_dataloader()
    elif eval_data == "val":
        df = dm.val_data.dataset_df[cols]
        dataloader = dm.val_dataloader()
    elif eval_data == "test":
        df = dm.test_data.dataset_df[cols]
        dataloader = dm.test_dataloader()

    labels = torch.unsqueeze(torch.tensor(df.y), 1)
    scores = torch.ones([df.shape[0],1])*torch.nan
    
    pl_trained_model.eval()
    with torch.no_grad():
        batch_start_idx = 0
        for batch in dataloader:
            #print(f"shape:{batch[0].shape}")
            nn_prob = (torch.sigmoid(pl_trained_model.model
                .forward(batch[0])))
            batch_end_idx = batch_start_idx + batch[0].shape[0]
            scores[batch_start_idx:batch_end_idx] = nn_prob
            batch_start_idx = batch_end_idx

    preds = (scores > 0.5)

    # Accuracy and AUC 
    acc = accuracy(preds, labels)
    auc = auroc(scores, labels, num_classes=2, pos_label=1)
    print(f'Accuracy: {acc}, AUC: {auc}')

#### Saving predictions to csv ####
    save_dict = {"model": model_name, "acc": acc.numpy(), "auc": auc.numpy()}
    if save_metrics:
        metric_df = pd.DataFrame(save_dict, index = pd.RangeIndex(1))
        metric_df.to_csv(f"{output_path}metrics.csv", index=False)
    if save_preds:
        preds_df = df.assign(
            nn_prob = scores.detach().numpy().squeeze(),
            nn_pred = nn_prob > 0.5
        )
        preds_df.to_csv(f"{output_path}predictions.csv", index=False)


    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')
