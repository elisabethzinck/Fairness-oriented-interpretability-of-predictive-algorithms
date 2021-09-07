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

#%% Load data
file_path = 'data\\processed\\german_credit_full.csv'
output_path = 'data\\processed\\german_credit_nn_pred.csv'
raw_data = pd.read_csv(file_path)

#%% Prepare data

X = raw_data.drop(['credit_score', 'person_id'], axis = 1)
X = one_hot_encode_mixed_data(X)
y = raw_data.credit_score
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X.shape[1]
n_output = 1
n_train = X_train.shape[0]

#%% Define dataloaders
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data[:, None] # Make vector into matrix
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], index
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(
    torch.FloatTensor(X_train), torch.FloatTensor(y_train))


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


# %% Training
net = Net(
    num_features = n_features, 
    num_hidden = n_hidden, 
    num_output = n_output)
optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)
criterion = nn.BCELoss()

trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
net.train()
losses = []
print(f'Epoch \t Loss \t Accuracy')
for e in range(EPOCHS):
    loss_epoch = 0.0
    output_epoch = np.empty(n_train, dtype = 'float')
    for i, data in enumerate(trainloader):
        X_batch, y_batch, index_batch = data
        #print(X_batch.shape[0])
        optimizer.zero_grad()

        output_batch = net(X_batch)
        loss = criterion(output_batch, y_batch)
        #print(output_batch[0:5,:])
        loss.backward()
        optimizer.step()

        output_epoch[index_batch] = output_batch.detach().numpy().squeeze()
        loss_epoch += loss

    yhat_epoch = output_epoch >= 0.5
    acc_epoch = accuracy_score(y_epoch, yhat_epoch)
    print(f'{e} \t{loss_epoch: .2f} \t{acc_epoch: .2f}')
    losses.append(loss_epoch)
plt.plot(losses)









# %%
