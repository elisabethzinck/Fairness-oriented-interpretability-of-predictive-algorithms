#%%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import myData

#%%

class ADNIDataModule(pl.LightningDataModule):
    def __init__(self, dataset = None):

        super().__init__()

        assert(dataset == 1 or dataset == 2)
        self.dataset = dataset # 1 or 2 (which field strength of MRI)
        self.time_horizon = '2y'

        self.id_var = 'rid'
        self.sens_vars = ['sex', 'age']
        self.y_var = 'y'

        self.label_to_y_map = {
            0: np.nan,          # Censoring
            1: 0,               # Normal controls
            2: 0,               # MCI deemed stable
            3: 1,               # AD
        }

        self.val_size = 0.2
        self.seed = 42

        self.batch_size = 32

        self.raw_data = self.load_raw_data()
        self.processed_data = self.process_raw_data()
        self.setup()

    def load_raw_data(self):
        raw_data = {}
        patient_groups = ['ad', 'mci', 'nc']
        for pat_grp in patient_groups:
            file_path = f'data/ADNI/raw/sorted_{pat_grp}{self.dataset}.csv'
            raw_data[pat_grp] = pd.read_csv(file_path)
            raw_data[pat_grp].columns = raw_data[pat_grp].columns.str.lower()
        return raw_data

    def process_raw_data(self):
        trainval_data = (
            pd.concat([self.raw_data['ad'], self.raw_data['nc']])
            .assign(y = lambda x: x.label.map(self.label_to_y_map))
            .set_index('rid')
            .drop(columns = 'label')
        )
        test_data = (
            self.raw_data['mci']
            .assign(y = lambda x: x[self.time_horizon].map(self.label_to_y_map))
            .set_index('rid')
            .drop(columns = ['label', '1y', '2y', '3y', '4y', '5y'])
            .dropna()
        )
        processed_data = {'trainval_data': trainval_data, 
                          'test_data': test_data}
        return processed_data

    def setup(self, stage = None):
        #Dividing data into X and y
        trainval_data = self.processed_data['trainval_data']
        X_trainval = trainval_data.drop(columns = 'y')
        y_trainval = trainval_data.y

        test_data = self.processed_data['test_data']
        X_test = test_data.drop(columns = 'y')
        y_test = test_data.y

        # One hot encode sex attribute 
        X_trainval = one_hot_encode_mixed_data(X_trainval)
        X_test = one_hot_encode_mixed_data(X_test)

        # Saving output dim and feature dim for plNet
        self.n_features = X_trainval.shape[1]
        self.n_output = 1

        #Split trainval into train and val
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X_trainval, 
            y_trainval, 
            X_trainval.index,
            test_size = self.val_size, 
            random_state = self.seed)

        # Saving rid/index of train val split
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = pd.Series(test_data.index)

        #Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

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


#%%
if __name__ == '__main__':
    dm = ADNIDataModule(dataset = 1)

# %%
