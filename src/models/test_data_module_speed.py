#%%
import time

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

if __name__ == '__main__':
    # Time 100 batches (from the validation set)
    print('Standard run')
    dm = CheXpertDataModule(
        **{"target_disease":"Cardiomegaly", 
        "uncertainty_approach": "U-Zeros",
        'augment_images': True})
    time_100_batches(dm)

    print('No image augmentation')
    dm = CheXpertDataModule(
        **{"target_disease":"Cardiomegaly", 
        "uncertainty_approach": "U-Zeros",
        'augment_images': False})
    time_100_batches(dm)



