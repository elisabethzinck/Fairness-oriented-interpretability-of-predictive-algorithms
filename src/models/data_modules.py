#%%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import myData

#%%
#######################################################################
class ADNIDataModule(pl.LightningDataModule):
    def __init__(self, dataset = None):

        super().__init__()

        assert(dataset == 1 or dataset == 2)
        self.dataset = dataset # 1 or 2 (which field strength of MRI)
        self.time_horizon = '2y'

        self.id_var = 'rid'
        self.sens_vars = ['sex', 'age']
        self.y_var = 'y'

        self.dataset_name = 'ADNI'+str(dataset)

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
            .assign(y = lambda x: x.label.map(self.label_to_y_map),
                    sex = lambda x: x.sex.map({'M':'Male', 'F':'Female'}))
            .set_index('rid')
            .drop(columns = 'label')
        )
        test_data = (
            self.raw_data['mci']
            .assign(y = lambda x: x[self.time_horizon].map(self.label_to_y_map),
                    sex = lambda x: x.sex.map({'M':'Male', 'F':'Female'}))
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

class CatalanDataModule(pl.LightningDataModule):
    def __init__(self, fold = None):
        super().__init__()
        self.file_path = 'data/processed/catalan-juvenile-recidivism/catalan-juvenile-recidivism-subset.csv'
        self.batch_size = 32 
        self.fold = fold
        self.seed = 42
        self.test_size = 0.2 
        self.kf = KFold(n_splits=5, shuffle = True, random_state=self.seed)  

        self.id_var = 'id'
        self.sens_vars = [
            'V1_sex', 'V8_age', 
            'V4_area_origin', 'V6_province'
            ]   
        self.y_var = 'V115_RECID2015_recid' 
        self.dataset_name = 'Catalan Recidivism'  

        self.load_raw_data()
        self.setup()
        if fold is not None:
            self.make_KFold_split(fold = fold)

    def load_raw_data(self):
        self.raw_data = pd.read_csv(self.file_path)
    
    def setup(self, stage = None):
        #One hot encoding 
        X = self.raw_data.drop(['V115_RECID2015_recid', 'id'], axis = 1)
        self.X = one_hot_encode_mixed_data(X)
        self.y = self.raw_data.V115_RECID2015_recid.to_numpy()
        
        # Saving output and features for plNet
        self.n_obs = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_output = 1

    def make_KFold_split(self, fold, stage = None):
        #splitting data into 5 folds and extracting desired fold_idx
        self.fold = fold
        splits = self.kf.split(self.X)
        train_val_idx, test_idx = next(x for i,x in enumerate(splits) if i==self.fold)
        
        # Partitioning data into train_val and test
        X_train_val = self.X.iloc[train_val_idx]
        y_train_val = self.y[train_val_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

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
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.test_data, batch_size=self.batch_size)

class GermanDataModule(pl.LightningDataModule):
    def __init__(self, fold = None):
        super().__init__()
        self.file_path = 'data/processed/german_credit_full.csv'
        self.batch_size = 32 
        self.seed = 42
        self.test_size = 0.2
        self.kf = KFold(n_splits=5, shuffle = True, random_state=self.seed)
        self.fold = fold   

        self.id_var = 'person_id'
        self.sens_vars = ['sex', 'age']   
        self.y_var = 'credit_score'
        self.dataset_name = 'German Credit'

        self.raw_data = self.load_raw_data()
        self.setup()
        if fold is not None:
            self.make_KFold_split(fold = fold)

    def load_raw_data(self):
        raw_data = pd.read_csv(self.file_path)
        return raw_data
    
    def setup(self, stage = None):
        #One hot encoding 
        self.y = self.raw_data['credit_score']
        X = self.raw_data.drop(['credit_score', 'person_id'], axis = 1)
        self.X = one_hot_encode_mixed_data(X)
        
        # Saving output and features for plNet
        self.n_obs = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_output = 1

    def make_KFold_split(self, fold, stage = None):
        #splitting data into 5 folds and extracting desired fold_idx
        self.fold = fold
        splits = self.kf.split(self.X)
        train_val_idx, test_idx = next(x for i,x in enumerate(splits) if i==self.fold)
        
        # Partitioning data into train_val and test
        X_train_val = self.X.iloc[train_val_idx]
        y_train_val = self.y[train_val_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

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
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        assert self.fold is not None, "Please specify a fold before using dataloader"
        return DataLoader(self.test_data, batch_size=self.batch_size)

class TaiwaneseDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.file_path = 'data/processed/taiwanese_credit.csv'

        self.batch_size = 32 
        self.test_size = 0.2
        self.val_size = 0.2 # Relative to train val
        self.seed = 42

        self.id_var = 'id'
        self.sens_vars = ['sex', 'age']
        self.y_var = 'default_next_month'
        self.dataset_name = 'Taiwanese Credit'

        self.load_raw_data()
        self.setup()

    def load_raw_data(self):
        self.raw_data = pd.read_csv(self.file_path)
    
    def setup(self, stage = None):
        #One hot encoding 
        X = self.raw_data.drop(columns = [self.y_var, self.id_var], axis = 1)
        X = one_hot_encode_mixed_data(X)
        y = self.raw_data[self.y_var].to_numpy()
        
        # Saving output and features for plNet
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_output = 1

        # Split data into train+validation and test
        all_idx = np.arange(self.n_obs) 
        train_val_idx, test_idx = train_test_split(
            all_idx, 
            test_size = self.test_size, 
            random_state = self.seed)
        X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

    
        # Split train+val into train and val
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X_train_val, 
            y_train_val, 
            train_val_idx,
            test_size = self.val_size, 
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
    
class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.folder_path = 'data/CheXpert/raw/'

        self.batch_size = 16
        #self.test_size = 0.2
        #self.val_size = 0.2 # Relative to train val
        self.seed = 42

        #self.id_var = 'id'
        #self.sens_vars = ['sex', 'age']
        #self.y_var = 'default_next_month' # Maybe depending on which patology we predict
        self.dataset_name = 'CheXpert'

        self.setup()

    
    def setup(self, stage = None):
        
        self.train_df = pd.read_csv(self.folder_path + 'CheXpert-v1.0-small/train.csv')
        self.val_df = pd.read_csv(self.folder_path + 'CheXpert-v1.0-small/valid.csv')
        
        # Saving n output and features for plNet
        #self.n_obs = X.shape[0]
        #self.n_features = X.shape[1]
        #self.n_output = 1

        # Split data into train+validation and test

    
        # Split train+val into train and val
       
        #Scaler to standardize for optimization step

        # Saving test, train and val idx in self


        if stage in (None, "fit"): 
            pass
        
        if stage in (None, "test"):
            pass

    def load_image(path):
        image_path = self.folder_path + path
        
    
    def train_dataloader(self):
        #return DataLoader(self.train_data, batch_size=self.batch_size)
        return None

    def val_dataloader(self):
        #return DataLoader(self.val_data, batch_size=self.batch_size)
        return None
    
    def test_dataloader(self):
        #return DataLoader(self.test_data, batch_size=self.batch_size)
        return None

    def predict_dataloader(self):
        #return DataLoader(self.test_data, batch_size=self.batch_size)
        return None
#%%
if __name__ == '__main__':
    #dm_adni = ADNIDataModule(dataset = 1)
    #dm_taiwan = TaiwaneseDataModule()
    #dm_german = GermanDataModule()
    #dm_german.make_KFold_split(fold = 2)
    #dm_catalan = CatalanDataModule()
    #dm_catalan.make_KFold_split(fold = 1)
    dm = CheXpertDataModule()

# %%
