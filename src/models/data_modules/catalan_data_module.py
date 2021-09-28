#%% Imports 
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import myData

#%% 
class CatalanDataModule(pl.LightningDataModule):
    def __init__(self, fold):
        super().__init__()
        self.file_path = 'data/processed/catalan-juvenile-recidivism/catalan-juvenile-recidivism-subset.csv'
        self.batch_size = 32 
        self.fold = fold
        self.seed = 42
        self.test_size = 0.2 
        self.kf = KFold(n_splits=5, shuffle = True, random_state=self.seed)     

        self.load_raw_data()
        self.setup()

    def load_raw_data(self):
        self.raw_data = pd.read_csv(self.file_path)
    
    def setup(self, stage = None):
        #One hot encoding 
        X = self.raw_data.drop(['V115_RECID2015_recid', 'id'], axis = 1)
        X = one_hot_encode_mixed_data(X)
        y = self.raw_data.V115_RECID2015_recid.to_numpy()
        
        # Saving output and features for plNet
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_output = 1

        #splitting data into 5 folds and extracting desired fold_idx
        splits = self.kf.split(X)
        train_val_idx, test_idx = next(x for i,x in enumerate(splits) if i==self.fold)
        
        # Partitioning data into train_val and test
        X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X_train_val, 
            y_train_val, 
            train_val_idx,
            test_size = self.test_size,
            random_state = self.seed)
       
        #Scaler to standardize for optimization step
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Saving test, train and val idx in self
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        if stage in (None, "fit"): 
            self.train_data = myData(X_train, y_train)
            self.val_data = myData(X_val, y_val)
        
        if stage in (None, "test"):
            self.test_data = myData(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# %%
