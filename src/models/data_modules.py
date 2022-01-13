#%%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize

from src.data.general_preprocess_functions import one_hot_encode_mixed_data
from src.models.general_modelling_functions import myData
from imgaug import augmenters as iaa

#%%
#######################################################################
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
        self.file_path = 'data/processed/german_credit.csv'
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

class CheXpertDataset(Dataset):
    def __init__(self, dataset_df, image_size, multi_label, target_disease,
     extended_image_augmentation, uncertainty_approach, simple_image_augmentation):
        """Dataset for CheXpert data
        
        Args:
            dataset_df (pd.DataFrame): DataFrame containing paths to images and targets. 
            image_size (tuple): image size in a tuple (height, width)
            extended_image_augmentation (bool): whether to augment images in an extended manner
            simple_image_augmentation (bool): whether to augment images in a simple manner
        """
        self.dataset_df = dataset_df
        self.image_size = image_size
        self.multi_label = multi_label
        self.target_disease = target_disease
        self.uncertainty_approach = uncertainty_approach
        self.extended_image_augmentation = extended_image_augmentation
        self.simple_image_augmentation = simple_image_augmentation

        if min(self.image_size) < 224:
            raise ValueError('DenseNet in Pytorch requires height and width to be at least 224')

        if simple_image_augmentation: 
            print("---- Initializing Simple Image Augmentation ----")
            self.augmenter = T.Compose([
                T.RandomHorizontalFlip(p=0.5)])
        if extended_image_augmentation: 
            print("---- Initializing Extended Image Augmentation ----")
            self.augmenter = T.Compose([
                T.RandomHorizontalFlip(p=0.25),
                T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale = (0.9, 1.1))], p=0.25),
                T.RandomApply(transforms=[T.RandomAdjustSharpness(sharpness_factor=2)], p=0.25),
                T.RandomApply(transforms=[T.RandomRotation(degrees=15)], p = 0.25)
            ]) 

        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        self.setup()

    def setup(self):
        self.N = self.dataset_df.shape[0]
        self.X_paths = self.dataset_df.Path

        # Uncertainty approach
        if self.uncertainty_approach == 'U-Ones':
            target_map = {
                np.nan: 0,  # unmentioned
                0.0: 0,     # negative
                -1.0: 1,    # uncertain
                1.0: 1      # positive
                }
        elif self.uncertainty_approach == 'U-Zeros':
            target_map = {
                np.nan: 0,  # unmentioned
                0.0: 0,     # negative
                -1.0: 0,    # uncertain
                1.0: 1      # positive
                }
        else:
            raise ValueError('Only uncertainty approaches U-Ones and U-Zeros are implemented.')

        # Create y matrix
        if self.multi_label:
            self.num_classes = len(self.labels)
            y = np.zeros((self.N, self.num_classes), dtype = int)
            for i, label in enumerate(self.labels):
                y[:,i] = self.dataset_df[label].map(target_map)
            self.y = y
        elif self.target_disease is not None:
            self.num_classes = 1
            y = self.dataset_df[self.target_disease].map(target_map)
            self.y = np.expand_dims(y, axis = 1)
        else:
            raise ValueError(
                'If multi_label is False, target disease must not be None')


    def __getitem__(self, idx):
        image_path = self.X_paths[idx]
        batch_x = self.load_image(image_path)
        batch_x = np.moveaxis(batch_x, source=-1, destination=0) #dim=(C, H, W)
        batch_y = self.y[idx]

        return batch_x, batch_y
        
    def __len__ (self):
        return self.N

    def load_image(self, image_path):
        """Loads image and turns it into array of appropiate size
        """
        image = Image.open(image_path)
        
        # Extended image augmentation 
        if self.extended_image_augmentation or self.simple_image_augmentation:
            image = self.augmenter(image)

        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.image_size)

        # Standardize image to match imagenet
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - imagenet_mean) / imagenet_std
        return image_array
    
    
    
class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.folder_path = 'data/CheXpert/raw/'
        self.dataset_name = 'CheXpert'

        self.uncertainty_approach = kwargs.get(
            'uncertainty_approach', 'U-Zeros')
        self.multi_label = kwargs.get('multi_label', False)
        self.target_disease = kwargs.get('target_disease', None)
        self.extended_image_augmentation = kwargs.get('extended_image_augmentation', False)
        self.simple_image_augmentation = kwargs.get('simple_image_augmentation', False)
        self.num_workers = kwargs.get('num_workers', 0)
        self.tiny_sample_data = kwargs.get('tiny_sample_data', False)

        self.batch_size = 32 
        self.image_size = (224, 224)
        self.test_size = 0.2
        self.val_size = 0.2 # Relative to train_val
        self.seed = 42

        self.setup()

    def create_dataset(self, patient_ids):
        df = patient_ids.merge(
                self.dataset_df, how = 'inner', on = 'patient_id')
        dataset = CheXpertDataset(
                df, 
                target_disease = self.target_disease,
                multi_label = self.multi_label,
                uncertainty_approach = self.uncertainty_approach,
                image_size = self.image_size,
                extended_image_augmentation = self.extended_image_augmentation, 
                simple_image_augmentation=self.simple_image_augmentation)
        return dataset
        
    def setup(self, stage = None):
        
        self.train_raw = pd.read_csv(self.folder_path + 'CheXpert-v1.0-small/train.csv')
        self.val_raw = pd.read_csv(self.folder_path + 'CheXpert-v1.0-small/valid.csv')

        if self.tiny_sample_data:
            dataset_df = self.val_raw
        else:
            dataset_df = self.train_raw
        
        self.dataset_df = (dataset_df
            .assign(
                patient_id = lambda x: x.Path.str.split('/').str[2],
                Path = lambda x: self.folder_path + x.Path)
            .loc[lambda x: x['Frontal/Lateral'] == 'Frontal']
            .query('Sex == "Male" or Sex == "Female"')
            )


        # Make splits based on patients
        patients = self.dataset_df[['patient_id']].drop_duplicates()
        train_val_patients, test_patients = train_test_split(
            patients, 
            test_size = self.test_size, 
            random_state = self.seed)
        train_patients, val_patients = train_test_split(
            train_val_patients, 
            test_size = self.val_size, 
            random_state = self.seed)


        if stage in (None, "fit"): 
            self.train_data = self.create_dataset(train_patients)
            self.val_data = self.create_dataset(val_patients)
        
        if stage in (None, "test"):
            self.test_data = self.create_dataset(test_patients)
        
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size = self.batch_size, 
            drop_last = False,
            num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            drop_last=False,
            num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            drop_last=False,
            num_workers = self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            drop_last=False,
            num_workers = self.num_workers)
#%%
if __name__ == '__main__':
    #dm_taiwan = TaiwaneseDataModule()
    #dm_german = GermanDataModule()
    #dm_german.make_KFold_split(fold = 2)
    #dm_catalan = CatalanDataModule()
    #dm_catalan.make_KFold_split(fold = 1)
    kwargs = {
       # "target_disease":"Cardiomegaly", 
        "multi_label": True,
        "uncertainty_approach": "U-Zeros",
        "tiny_sample_data": True, 
        "extended_image_augmentation": False}
    dm = CheXpertDataModule(**kwargs)
    X, y = next(iter(dm.train_dataloader()))
    print(X.shape)
    print(y.shape)

    plt.imshow(X[0].permute(1, 2, 0))

    #dm_t = TaiwaneseDataModule()
    #tmp_t = next(iter(dm_t.train_dataloader()))

# %%
