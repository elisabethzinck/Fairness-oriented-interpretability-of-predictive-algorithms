#%%
import pandas as pd

from src.models.data_modules import *
from biasbalancer.utils import label_case

# %%

# Train sizes
def get_size(dm, which):
    attrname = which+'_idx'
    idx = getattr(dm, attrname)
    return len(idx)

def get_dataset_info(dm):
    index = [0]

    if hasattr(dm, 'fold'):
        n_folds = dm.kf.get_n_splits()
        df = []
        for i in range(n_folds):

            dm.make_KFold_split(fold = i)

            tmp = pd.DataFrame({
                'dataset': dm.dataset_name,
                'fold': str(i),
                'train_size': get_size(dm, 'train'),
                'val_size': get_size(dm, 'val'),
                'test_size': get_size(dm, 'test'),
            }, index = index)

            df.append(tmp)
        
        res = pd.concat(df).reset_index(drop = True)

        
    else:
        res = pd.DataFrame({
            'dataset': dm.dataset_name,
            'fold': ' ',
            'train_size': get_size(dm, 'train'),
            'val_size': get_size(dm, 'val'),
            'test_size': get_size(dm, 'test'),
        }, index = index)

    hyperparams = get_hyperparameter_info(dm)
    res = pd.concat([res, hyperparams], axis = 1)
    return res

def get_hyperparameter_info(dm):
    hyper_paths = {
        'German Credit': 'data/predictions/german_credit_nn_pred_hyperparams',
        'Catalan Recidivism': 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred_hyperparams',
        'Taiwanese Credit': 'data/predictions/taiwanese_nn_pred_hyperparams'}
    

    hyperpath = hyper_paths[dm.dataset_name] + '.csv'

    hyperdf = pd.read_csv(hyperpath)

    def hidden_to_string(col):
        col = col.fillna(0)
        col = col.map(int)
        col = col.map(str)
        col = col.replace('0', '')
        return col

    # Clean up n_hidden column
    n_hidden_cols = sorted([col for col in hyperdf.columns if col.startswith('n_hidden')])
    for col in n_hidden_cols:
        hyperdf[col] = hidden_to_string(hyperdf[col])

    if len(n_hidden_cols) == 1:
        hyperdf['n_hidden'] = '(' + hyperdf['n_hidden_0'] + ')'
    elif len(n_hidden_cols) == 2:
        hyperdf['n_hidden'] = '(' + hyperdf['n_hidden_0'] + ', ' + hyperdf['n_hidden_1'].map(str) + ')'
    elif len(n_hidden_cols) == 3:
        hyperdf['n_hidden'] = '(' + hyperdf['n_hidden_0'] \
            + ', ' + hyperdf['n_hidden_1'] \
            + ', ' + hyperdf['n_hidden_2'] + ')'
    else:
        raise ValueError('Woops, this is not implemented for n_layers > 3')
    
    hyperdf.drop(columns = n_hidden_cols, inplace = True)

    return hyperdf

# %%
datamodules = [
    GermanDataModule(),
    TaiwaneseDataModule(),
    CatalanDataModule()]

# Getting the data
datasetinfo = pd.concat(
    [get_dataset_info(dm) for dm in datamodules]
    ).reset_index(drop = True)
datasetinfo


# Making it pretty for latex
datasetinfo['lr'] = datasetinfo['lr'].round(4)
datasetinfo['p_dropout'] = datasetinfo['p_dropout'].round(2)

datasetinfo.columns = [label_case(col) for col in datasetinfo.columns]
print(datasetinfo.to_latex(index = False))



# %%
