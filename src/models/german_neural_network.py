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

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import logging
logging.getLogger("lightning").setLevel(logging.ERROR) # Do not see warnings and suggestions

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
    def __init__(self, num_features, num_hidden, num_output, p_dropout = 0):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden), 
            nn.Dropout(p = p_dropout),
            nn.ReLU(), 
            nn.Linear(num_hidden, num_output),
        )


    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x) 

        return x

test_net = Net(
    num_features = n_features, 
    num_hidden = 10, 
    num_output = n_output,
    p_dropout = 0.5)
print(test_net)

# Check net
# See all parameters
#print(list(net.named_parameters()))
# Total size of parameters
n_params = get_n_total_parameters(test_net)
test_observation = torch.randn(5, n_features)
test_output = test_net(test_observation)

#%% Training using lightning
# pytorch lightning net
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

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)

#%% Training
net = Net(
    num_features = n_features, 
    num_hidden = 10, 
    num_output = n_output)
plnet = BinaryClassificationTask(model = net, lr = 1e-3)
train_data = myData(X_train, y_train)
test_data = myData(X_test, y_test)
trainloader = DataLoader(dataset=train_data, batch_size=32)
testloader = DataLoader(dataset=test_data, batch_size = 32)

early_stopping = EarlyStopping('val_loss', patience = 3)
trainer = pl.Trainer(
    fast_dev_run = False,
    log_every_n_steps = 1, 
    callbacks = [early_stopping])
trainer.fit(plnet, trainloader, testloader)

#%% Optimizing using optuna
def objective_function(trial: optuna.trial.Trial):
    # Define parameters
    n_hidden = trial.suggest_int('n_hidden', 1, 100)
    hyperparameters = {'n_hidden': n_hidden}
    lr = trial.suggest_loguniform('lr', 1e-5, 1e2)
    p_dropout = trial.suggest_uniform('p_dropout', 0, 0.5)
    
    # Define network and lightning
    net = Net(
        num_features = n_features, 
        num_hidden = n_hidden, 
        num_output = n_output,
        p_dropout = p_dropout)
    plnet = BinaryClassificationTask(model = net, lr = lr)

    # Define data
    train_data = myData(X_train, y_train)
    test_data = myData(X_test, y_test)
    trainloader = DataLoader(dataset=train_data, batch_size=32)
    testloader = DataLoader(dataset=test_data, batch_size = 32)

    early_stopping = EarlyStopping('val_loss', patience = 3)
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = pl.Trainer(
        fast_dev_run = False,
        log_every_n_steps = 1, 
        max_epochs = 50,
        callbacks = [early_stopping, optuna_pruning])

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(plnet, trainloader, testloader)

    return trainer.callback_metrics['val_loss'].item()

study = optuna.create_study(direction = 'minimize')
n_minutes = 2
study.optimize(objective_function, timeout = n_minutes*60)
#%%
best_trial = study.best_trial
all_trials = study.trials
all_params = [t.params for t in all_trials]


#%%
optuna.visualization.plot_optimization_history(study)

#%%
optuna.visualization.plot_intermediate_values(study)

#%%
optuna.visualization.plot_param_importances(study)

#%%
fig = optuna.visualization.matplotlib.plot_intermediate_values(study)
fig.show()




# To open tensorboard (type in command line)
# tensorboard --logdir lightning_logs





# %%
