#%%
import time
import platform
import os

from src.models.data_modules import CheXpertDataModule
from src.models.general_modelling_functions import print_timing
#%%

def time_100_batches(dm):
    loader = dm.val_dataloader()
    print(f'n_batches = {len(loader)}')
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(i)
        if i == 100:
            break
    t1 = time.time()
    print_timing(t0, t1, 'Time to loop through batches')

def time_val_batches(dm):
    loader = dm.val_dataloader()
    print(f'n_batches = {len(loader)}')
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print(i)
    t1 = time.time()
    print_timing(t0, t1, 'Time to loop through val batches')

def time_train_batches(dm):
    loader = dm.train_dataloader()
    print(f'n_batches = {len(loader)}')
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print(i)
    t1 = time.time()
    print_timing(t0, t1, 'Time to loop through train batches')



if __name__ == '__main__':
    # If running on linux and > 2 cpus are available, also test with num workser
    if platform.system() == 'Linux':
        n_avail_cpus = len(os.sched_getaffinity(0))
        print(f'I am on the HPC. Available number of CPUS:{n_avail_cpus}')
        if n_avail_cpus > 2:
            n_workers = min(8, n_avail_cpus-1)
            print('With image aug')
            dm = CheXpertDataModule(
                **{"target_disease":"Cardiomegaly", 
                "uncertainty_approach": "U-Zeros",
                'num_workers': n_workers,
                'augment_images': True})
            time_train_batches(dm)

            print('Without image aug')
            dm = CheXpertDataModule(
                **{"target_disease":"Cardiomegaly", 
                "uncertainty_approach": "U-Zeros",
                'num_workers': n_workers,
                'augment_images': False})
            time_train_batches(dm)





