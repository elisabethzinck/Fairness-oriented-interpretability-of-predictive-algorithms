import torch
import os
import pandas as pd

from torchmetrics.functional import accuracy, auroc
from src.models.data_modules import CheXpertDataModule
from src.models.cheXpert_modelling_functions import BinaryClassificationTaskCheXpert
from src.models.data_modules import CheXpertDataModule

#### Setup #######
eval_data = "val"
model_name = "test_model"
model_type = "last"
model_ckpt = f"models/CheXpert/checkpoints_from_trainer/{model_name}/{model_type}.ckpt"
dm = CheXpertDataModule(**{"target_disease":"Cardiomegaly", "uncertainty_approach": "U-Zeros"})

output_path = f"data/CheXpert/predictions/{model_name}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

save_metrics = False 
save_preds = False

# input checks
assert eval_data in ['train', 'val', 'test'], "eval_data must be 'train', 'val' or 'test'"
assert model_type in ['best', 'last'], "model_type must be 'last' or 'best'"

#### Loading Checkpointed Model #######
model = torch.hub.load(
    'pytorch/vision:v0.10.0', 
    'densenet121', 
    pretrained = True)
in_features_classifier = model.classifier.in_features
model.classifier = torch.nn.Linear(in_features_classifier, 1)
pl_model = BinaryClassificationTaskCheXpert(model = model)
print(f"Bias of ImageNet DenseNet:\n{pl_model.model.classifier.bias}")

pl_trained_model = pl_model.load_from_checkpoint(model_ckpt)
print(f"Bias of {model_name}:\n{pl_trained_model.model.classifier.bias}")

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

pl_trained_model.model.eval()
batch_start_idx = 0
for batch in dataloader:
    print(f"shape:{batch[0].shape}")
    nn_prob = (torch.sigmoid(pl_trained_model.model
        .forward(batch[0])))
    batch_end_idx = batch_start_idx + batch[0].shape[0]
    scores[batch_start_idx:batch_end_idx] = nn_prob
    batch_start_idx = batch_end_idx

preds = (scores > 0.5)

# Accuracy and AUC 
acc = accuracy(preds, labels)
auc = auroc(scores, labels, num_classes=2, pos_label=1)

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


