#%%
import pandas as pd
import numpy as np
import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# %%

def get_best_version(folder_path):
    """Return tb_logger of the best version (based on best val_loss)
    Params:
        folder_path (str): Folder with in which version_* folders are located
    
    Returns:
        tensorboard logger object (EventAccumulator) from best version
    
    """
    version_list = os.listdir(folder_path)
    if len(version_list) == 1:
        tb_best = EventAccumulator(f'{folder_path}/{version_list[0]}')
        tb_best.Reload()
    else:
        best_val_loss = 10000000
        for v in version_list:
            tb_log = EventAccumulator(f'{folder_path}/{v}')
            tb_log.Reload()
            val_loss = get_metric(tb_log, 'val_loss')
            if val_loss < best_val_loss:
                tb_best = tb_log
    return tb_best

def get_metric(tb_log, metric, verbose = False):
    """Return metric of the best model (based on val_loss)"""
    metrics_logged = tb_log.Tags()['scalars']
    if metric in metrics_logged:
        # Format [wall_time, step, value]
        _, _, val_loss = zip(*tb_log.Scalars('val_loss')) 
        best_model_idx = np.argmin(val_loss)

        _, _, metric_values = zip(*tb_log.Scalars(metric))
        metric_best = metric_values[best_model_idx]
        metric_best = round(metric_best, 3)
    else:
        if verbose:
            print(f'Requested metric `{metric}` is not in logger with path `{tb_log.path}`. Returning None')
        metric_best = np.nan
    return metric_best

def get_weight_decay(model_name):
    "Extract weight decay from model name"
    match = re.findall('wd=(....)', model_name)
    if len(match) > 0:
        match = float(match[0])
    else:
        match = 0
    return match

def get_dropout(model_name):
    "Extract dropout from model name"
    match = re.findall('dp=(....)', model_name)
    if len(match) > 0:
        match = float(match[0])
    else:
        match = 0
    return match

def get_imgaug(model_name):
    "Extract image augmentation from model name"
    imgaug_match = re.search('imgaug', model_name)
    simple_match = re.search('simple', model_name)
    if simple_match or model_name == 'benchmark_adam':
        return 'Simple'
    elif imgaug_match:
        return 'Extensive'
    else:
        return 'None'



# %%
if __name__ == '__main__':
    folder_path = 'models/CheXpert/lightning_logs/'
    model_paths =  [
        "adam",
        "adam_wd=1e-2",
        "adam_wd=1e-3",
        "adam_dp=2e-1",
        "adam_wd=1e-2_dp=2e-1",
        "adam_wd=1e-3_dp=2e-1",
        "adam_dp=4e-1",
        "adam_wd=1e-2_dp=4e-1",
        "adam_wd=1e-3_dp=4e-1",
        "adam_imgaug",
        "adam_wd=1e-2_imgaug",
        "adam_wd=1e-3_imgaug",
        "adam_dp=2e-1_imgaug",
        "adam_wd=1e-2_dp=2e-1_imgaug",
        "adam_wd=1e-3_dp=2e-1_imgaug",
        "adam_dp=4e-1_imgaug",
        "adam_wd=1e-2_dp=4e-1_imgaug",
        "adam_wd=1e-3_dp=4e-1_imgaug",
        "adam_imgaug_simple",
        "adam_wd=1e-2_imgaug_simple",
        "adam_wd=1e-3_imgaug_simple",
        "adam_dp=2e-1_imgaug_simple",
        "adam_wd=1e-2_dp=2e-1_imgaug_simple",
        "adam_wd=1e-3_dp=2e-1_imgaug_simple",
        "adam_dp=4e-1_imgaug_simple",
        "adam_wd=1e-2_dp=4e-1_imgaug_simple",
        "adam_wd=1e-3_dp=4e-1_imgaug_simple"]

    print(f'Number of models: {len(model_paths)}')

    df_rows = []
    for mod_path in model_paths:
        tb_log = get_best_version(f'{folder_path}{mod_path}')
        row = pd.DataFrame({
            'path': mod_path, 
            'imgaug': get_imgaug(mod_path),
            'dropout': get_dropout(mod_path),
            'weight_decay': get_weight_decay(mod_path),
            'val_auroc': get_metric(tb_log, 'val_auroc'),
            'val_loss': get_metric(tb_log, 'val_loss'),
            'train_auroc': get_metric(tb_log, 'train_auroc'),
            'train_loss': get_metric(tb_log, 'train_loss')},
            index = [0])
        df_rows.append(row)
    df = (pd.concat(df_rows)
        .reset_index(drop = True)
        .drop(columns = ['path'])
        .sort_values(by = 'val_auroc', ascending = False))

    print(df.to_latex(index = False))

# %%

# %%
