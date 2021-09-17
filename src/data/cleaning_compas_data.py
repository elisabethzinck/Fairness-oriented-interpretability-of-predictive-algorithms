from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw_file_path = 'data\\raw\\compas\\compas-scores-two-years.csv'
    processed_file_path = 'data\\processed\\compas\\compas-scores-two-years-subset.csv'
    processed_file_path_preds = 'data\\processed\\compas\\compas-scores-two-years-pred.csv'
    
    compas_raw = pd.read_csv(raw_file_path)
    compas_raw.head()

    # We need to filter out rows where **days_b_screening_arrest** is over 30 or under -30
    # See article \cite{Larson2016} for info
    compas = compas_raw.loc[(compas_raw['days_b_screening_arrest'] <= 30) & (compas_raw['days_b_screening_arrest'] >= -30)]
    assert(compas.shape[0] == 6172)

    #Omitting the 'other' category.
    # We also omit Asian (31 obs) and Native American (11) obs.
    compas.groupby('race').agg({'id': 'count'})
    compas_subset = compas.loc[compas.race.isin(['African-American', 'Caucasian', 'Hispanic'])]

    # Dropping irrelevant columns and writing to csv 
    compas_subset = compas_subset[compas_subset.columns.difference(['violent_recid', 'decile_score.1'])]
    compas_subset.to_csv(processed_file_path, index = False)

    # Data frame with predictions 
    pred = compas_subset[['id', 'sex', 'age', 'age_cat', 'race', 
                         'decile_score','score_text', 
                         'v_decile_score', 'v_score_text',
                         'two_year_recid']]
    
    pred = pred.assign(
        pred_high = (pred.score_text == 'High'), 
        v_pred_high = (pred.v_score_text == 'High'),
        pred_medium_high = (pred.score_text != 'Low'),
        v_pred_medium_high = (pred.v_score_text != 'Low'))
    print(pred.columns)
    
    pred.to_csv(processed_file_path_preds, index=False)            