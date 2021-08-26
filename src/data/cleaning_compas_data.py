import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw_file_path = 'data\\raw\\compas-scores-two-years.csv'
    processed_file_path = 'data\\processed\\compas-scores-two-years-subset.csv'

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

    compas_subset[compas_subset.columns.isnan()]

    # F5: 1 obs --> sexual abuse of minor by adult 
    # CO3: 2 obs --> drinking on/near licensed establishment 
    # MO3: 30 obs --> trespassing or drinking uncovered alc. beverage 
    # F6: 3 obs Capital Felony --> Murder in the first degree
    # F7: 17 obs Armed Kidnapping, Burglary dwelling Armed 

    cols = ['r_charge_degree', 'r_charge_desc',\
            'vr_charge_degree', 'vr_charge_desc']

    MO3_frame = (compas_subset[
        (compas_subset.r_charge_degree == '(MO3)') |
        (compas_subset.vr_charge_degree == '(MO3)')
        ])

    F6_frame = (compas_subset[
        (compas_subset.r_charge_degree == '(F6)') |
        (compas_subset.vr_charge_degree == '(F6)')
        ])

    F5_frame = (compas_subset[
        (compas_subset.r_charge_degree == '(F5)') |
        (compas_subset.vr_charge_degree == '(F5)')
        ])

    F7_frame = (compas_subset[
        (compas_subset.r_charge_degree == '(F7)') |
        (compas_subset.vr_charge_degree == '(F7)')
        ])