import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw_file_path = 'data/raw/compas/compas-scores-two-years.csv'
    compas = pd.read_csv(raw_file_path)
    compas.head()
    folder = "data/processed/compas/"

    # We need only choose the rows where **days_b_screening_arrest** is between -30 and 30
    # See article \cite{Larson2016} for info
    (compas.query
        ("days_b_screening_arrest<=30 and days_b_screening_arrest>=-30",
        inplace = True))
    assert(compas.shape[0] == 6172)
    #output file paths
    processed_file_path_preds = f'{folder}compas-scores-two-years-pred.csv'

    #subsetting 'African-American' and 'Caucasian'
    (compas.query
        ("race=='African-American' or race=='Caucasian'",
        inplace=True)
    )

    # Filtering away observations not followed for 2 years
    compas.query("screening_date <= '2014-04-01'", inplace = True)

    # Dropping irrelevant columns and writing to csv 
    compas.drop(columns = ['violent_recid', 'decile_score.1'], inplace=True)

    # Data frame with predictions 
    predictions = compas[['id', 'sex', 'age', 'age_cat', 'race', 
                   'decile_score','score_text', 'two_year_recid']]
    
    predictions = predictions.assign(
        pred = (predictions.score_text != 'Low'))

    predictions.to_csv(processed_file_path_preds, index=False)   
