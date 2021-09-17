#%% Imports
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger('lightning').setLevel(logging.ERROR)

#%% Helper Functions
def get_n_total_parameters(pytorch_model):
    n_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    return n_params

def get_n_hidden_list(params):
    """Extract list of hidden units from parameter dict"""
    n_hidden_list = []
    for i in range(params['n_layers']):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(params[name])
    return n_hidden_list


#%% Define dataset
class myData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data[:, None]) # Make vector into matrix
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

#%% Pytorch network
class Net(nn.Module):
    def __init__(self, num_features, num_hidden_list, num_output, p_dropout = 0):
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

#%% Pytorh Lightning network
class BinaryClassificationTask(pl.LightningModule):
    def __init__(self, model, lr = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

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
