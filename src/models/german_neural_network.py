#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import get_n_total_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

#%% Load data
file_path = 'data\\processed\\german_credit_full.csv'
output_path = 'data\\processed\\german_credit_nn_pred.csv'
raw_data = pd.read_csv(file_path)

#%% Prepare data

X = raw_data.drop(['credit_score', 'person_id'], axis = 1)
X = one_hot_encode_mixed_data(X)
y = raw_data.credit_score.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X.shape[1]
n_output = 1
n_train = X_train.shape[0]

#%% Define dataloaders
class myData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data[:, None] # Make vector into matrix
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


# %% Define network
class Net(nn.Module):
    def __init__(self, num_features, num_hidden, num_output):
        super(Net, self).__init__()

        # Layer 1
        self.l1 = nn.Linear(num_features, num_hidden)

        # Layer 2
        self.l2 = nn.Linear(num_hidden, num_output)


    def forward(self, x):
        # Layer 1
        x = self.l1(x)
        x = F.relu(x)

        # Layer 2
        x = self.l2(x)
        x = torch.sigmoid(x) 

        return x

net = Net(num_features = n_features, num_hidden = 10, num_output = n_output)
print(net)

# %% Check net
# See all parameters
#print(list(net.named_parameters()))
# Total size of parameters
n_params = get_n_total_parameters(net)
test_observation = torch.randn(5, n_features)
test_output = net(test_observation)

# %% Define things for training 
EPOCHS = 40
BATCH_SIZE = 100
LEARNING_RATE = 0.02

n_hidden = 10




#%% Training using lightning
# pytorch lightning net
class BinaryClassificationTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat, y)

        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_loss": loss, 'train_acc': acc}
        self.log_dict(metrics, on_epoch = True, on_step = False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, 'val_acc': acc}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, 'test_acc': acc}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat, y)
        y_hat_binary = y_hat >= 0.5
        acc = accuracy_score(y, y_hat_binary)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)


plnet = BinaryClassificationTask(net)
print(plnet)
#%%
train_data = myData(
    torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = myData(
    torch.FloatTensor(X_test), torch.FloatTensor(y_test))
trainloader = DataLoader(dataset=train_data, batch_size=32)
testloader = DataLoader(dataset=test_data, batch_size = 32)

early_stopping = EarlyStopping('val_loss', patience = 3)
trainer = pl.Trainer(
    log_every_n_steps = 1, 
    callbacks = [early_stopping])
trainer.fit(plnet, trainloader, testloader)








# %%
