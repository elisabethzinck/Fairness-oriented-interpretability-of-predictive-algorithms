import pandas as pd
from pathlib import Path
def get_compas_data(normalize_decile_scores=False):
    """Returns preprocessed compas data for notebook walktrough

    The function returns a pre-processed version of the COMPAS dataset used for the ProPublica article "Machine Bias" [ANGWIN2016]_. The dataset is pre-processed in the following way:

    * Subsetting to individuals with races African-American and Caucasian
    * Exclude individuals with ``days_b_screening_arrest`` :math:`>` 30 and ``days_b_screening_arrest`` :math:`<` -30 because data quality is questionable
    * Exclude individuals with ``screening_date`` > '2014-04-01' because these could not be followed for 2 years. 
        
    Args:
        normalize_decile_scores (bool): If true, the decile scores are normalized to be within [0,1] by dividing original scores by 10. 


    Returns: 
        DataFrame: data frame with preprocessed compas data. 
        Columns of the dataframe are ``id``, ``sex``, ``age``, ``age_cat``, ``race``, ``decile_score``, ``score_text``, ``two_year_recid``

    .. [ANGWIN2016] Angwin, J., Larson, J., Mattu, S., and Kirchner, L. (2016). Machine Bias. propublica.org.
    
    """
    root_path = Path(__file__).parent.parent
    raw_file_path = root_path / 'data/raw/compas-scores-two-years.csv'
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

    if normalize_decile_scores:
        df['decile_score'] = df['decile_score']/10

    return(df)

if __name__ == "__main__":
    df = get_compas_data()
    print(df.head())