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
        assert(dataset == 1 or dataset == 2)
        self.dataset = dataset # 1 or 2 (which field strength of MRI)
        self.time_horizon = '2y'

        self.id_var = 'rid'
        self.sens_vars = ['sex', 'age']
        self.y_var = 'y'

        label_to_y_map = {
            0: np.nan,          # Censoring
            1: 0,               # Normal controls
            2: 0,               # MCI deemed stable
            3: 1,               # AD
        }

        self.raw_data = self.load_raw_data()

    def load_raw_data(self):
        raw_data = {}
        patient_groups = ['ad', 'mci', 'nc']
        for pat_grp in patient_groups:
            file_path = f'data/ADNI/raw/sorted_{pat_grp}{self.dataset}.csv'
            raw_data[pat_grp] = pd.read_csv(file_path)
            raw_data[pat_grp].columns = raw_data[pat_grp].columns.str.lower()
        return raw_data

    def setup(self):
        # Gather training and validation data
        trainval_data = pd.concat([raw_data['ad'], raw_data['nc']])
        trainval_data[y_var] = trainval_data.label.map(label_to_y_map)

        # Clean test data
        test_data = raw_data['mci']
        test_data[y_var] = test_data[time_horizon].map(label_to_y_map)
        # To do: Figure out how to solve censoring

        # To do: Split trainval into train and val

        # To do: Standardize
#%%
if __name__ == '__main__':
    dm = ADNIDataModule(dataset = 1)


# %%
