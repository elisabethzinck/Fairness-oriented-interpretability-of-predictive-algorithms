#%% Imports
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

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

import warnings
warnings.simplefilter("ignore")

#%% Define dataset
class myData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data[:, None]) # Make vector into matrix
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


# %% Define network
class Net(nn.Module):
    def __init__(self, num_features, num_hidden_list,  num_output, p_dropout = 0):
        super(Net, self).__init__()
        n_hidden_layers = len(num_hidden_list)
        self.layers = []
        
        input_dim = num_features
        for i in range(n_hidden_layers):
            output_dim = num_hidden_list[i]
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.layers.append(nn.Dropout(p = p_dropout))
            self.layers.append(nn.ReLU())
            input_dim = output_dim
        
        # Last layer (without activation function)
        self.layers.append(nn.Linear(num_hidden_list[-1], num_output))
        self.layers.append(nn.Dropout(p = p_dropout))

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x) 

        return x
    
# Check net
#test_net = Net(
#    num_features = n_features, 
#    num_hidden = 10, 
#    num_output = n_output)
#print(test_net)
#test_observation = torch.randn(5, n_features)
#test_output = test_net(test_observation)


#%% Define lightning module
class BinaryClassificationTask(pl.LightningModule):
    def __init__(self, model, lr = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

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
        y_hat_binary = (y_hat >= 0.5)
        acc = accuracy_score(y, y_hat_binary)
        return loss, acc

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat.to_numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)


#%% Objective function for optuna optimizer
def objective_function(trial: optuna.trial.Trial):
    # Define parameters
    n_layers = trial.suggest_int('n_layers', 1, 2)
    n_hidden_list = []
    for i in range(n_layers):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(trial.suggest_int(name, 1, 10))
    lr = trial.suggest_loguniform('lr', 1e-5, 1e2)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.5)
    
    # Define network and lightning
    net = Net(
        num_features = n_features, 
        num_hidden_list = n_hidden_list, 
        num_output = n_output,
        p_dropout = p_dropout)
    plnet = BinaryClassificationTask(model = net, lr = lr)

    early_stopping = EarlyStopping('val_loss', patience = 3)
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, 
        max_epochs = 50,
        callbacks = [early_stopping, optuna_pruning], 
        deterministic = True)

    trainer.fit(plnet, train_loader, val_loader)

    return trainer.callback_metrics['val_loss'].item()


# To open tensorboard (type in command line)
# tensorboard --logdir lightning_logs
#%%

if __name__ == "__main__":
    pl.seed_everything(42)

    #Load data
    file_path = 'data\\processed\\german_credit_full.csv'
    output_path = 'data\\processed\\german_credit_nn_pred.csv'
    raw_data = pd.read_csv(file_path)

    #Prepare data
    X = raw_data.drop(['credit_score', 'person_id'], axis = 1)
    X = one_hot_encode_mixed_data(X)
    y = raw_data.credit_score.to_numpy()
    n_features = X.shape[1]
    n_output = 1

    # Make splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, 
        y_train_val, 
        test_size = 0.2, random_state = 42)

    # Standardize for optimization step
    scaler_optim_step = StandardScaler()
    X_train = scaler_optim_step.fit_transform(X_train)
    X_val = scaler_optim_step.transform(X_val)

    scaler_predict_step = StandardScaler()
    X_train_val = scaler_predict_step.fit_transform(X_train_val)
    X_test = scaler_predict_step.transform(X_test)

    # Define dataloaders
    train_loader = DataLoader(
        dataset=myData(X_train, y_train), batch_size=32)
    val_loader = DataLoader(
        dataset=myData(X_val, y_val), batch_size = 32)
    train_val_loader = DataLoader(
        dataset=myData(X_train_val, y_train_val), batch_size = 32)

    # Find optimal model
    study = optuna.create_study(
       direction = 'minimize', 
       sampler = TPESampler(seed=10))
    max_minutes = 1
    study.optimize(
        objective_function, 
        timeout = max_minutes*60)

    study.best_trial

#%% 
def get_n_hidden_list(params):
    n_hidden_list = []
    for i in range(params['n_layers']):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(params[name])
    return n_hidden_list

# %% Train model on all data
params = study.best_trial.params
n_hidden_list = get_n_hidden_list(params)

# Define network and lightning
net = Net(
    num_features = n_features, 
    num_hidden_list = n_hidden_list, 
    num_output = n_output,
    p_dropout = params['p_dropout'])
plnet = BinaryClassificationTask(model = net, lr = params['lr'])


trainer = pl.Trainer(
    fast_dev_run = False,
    log_every_n_steps = 1, 
    max_epochs = 50,
    deterministic = True)

trainer.fit(plnet, train_val_loader)
#%%
predictions = plnet.model.forward(torch.Tensor(X_test))
pred_binary = (predictions >= 0.5)
accuracy_score(y_test, pred_binary)
# %%
