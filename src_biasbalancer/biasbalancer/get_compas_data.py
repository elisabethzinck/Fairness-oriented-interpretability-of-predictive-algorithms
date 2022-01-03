import pandas as pd
def get_compas_data(normalize_decile_scores=False):
    """Returns preprocessed compas data for notebook walktrough
        
        Args:
            normalize_decile_scores (bool): Returns scores in [0,1]

        Returns: 
            data frame with preprocessed compas data. 
            Columns of the dataframe are 
            TODO fix columsn in a nice manner
            'id', 'sex', 'age', 'age_cat', 'race', 
                'decile_score','score_text', 'two_year_recid'
    """
    raw_file_path = 'src_biasbalancer/data/raw/compas-scores-two-years.csv'
    compas = pd.read_csv(raw_file_path)

    #Filter out rows where **days_b_screening_arrest** is over 30 or under -30
    (compas.query(
        "days_b_screening_arrest<=30 and days_b_screening_arrest>=-30",
        inplace = True)
        )
    #subsetting 'African-American' and 'Caucasian'
    (compas.query
        ("race=='African-American' or race=='Caucasian'",
        inplace=True)
        )
    # Filtering away observations not followed for 2 years
    compas.query("screening_date <= '2014-04-01'", inplace = True)

    # Data frame with predictions 
    df = compas[['id', 'sex', 'age', 'age_cat', 'race', 
                'decile_score','score_text', 'two_year_recid']]
    df = df.assign(pred = (df.score_text != 'Low'))

    return(df)

if __name__ == "__main__":
    df = get_compas_data()
    print(df.head())