#%% Importing
import pandas as pd

#%% Loading data
raw_file_path = 'data\\raw\\catalan-juvenile-recidivism\\reincidenciaJusticiaMenors.xlsx'
processed_file_path = 'data\\processed\\catalan-juvenile-recidivism'
column_translation_file_path = 'data\\raw\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-columns.xlsx'
    
raw = pd.read_excel(raw_file_path)
columns_translation = pd.read_excel(column_translation_file_path)

# Shape of data
#raw.shape # (4753, 143)

# renaming to english columns 
catalan_names = raw.columns
raw.columns = columns_translation.translation 
#%% Filling or dropping NaNs 

# Dropping columns with only NaN 
df = raw.drop(raw.columns[(raw.isnull().sum(axis = 0) == raw.shape[0])], axis = 1)

# Offenders with missing geographical areas of origin all come from Espanya
# so we impute Espanya on missing values 
df.loc[df.V4_area_origin.isnull()].V2_nationality_type.unique()
df.loc[df.V4_area_origin.isnull(), 'V4_area_origin'] = 'Espanya'

# For V12 and V20 NaN is equivalent to zero
df.V12_n_criminal_record.fillna(0, inplace = True)
df.V20_n_juvenile_records.fillna(0, inplace = True)

# For V26 we combine with column V25 because it has info about MRM and ATM, 
# which matches the places with NaNs in V26
df.V26_finished_measure_grouped.fillna(df.V25_MRM_ATM_or_enforcement_actions, inplace = True)

# Dropping rows with NaNs in V5, V6, V7 and V8
df.dropna(axis = 0, subset=['V5_age_cat', 'V6_province', 'V7_region', 'V8_age'], inplace = True)

#%% Filtering out columns, which are either specific MRM, ATM, Savry, 2013 or 2015 data
dfsub = df.filter(list(df.columns[list(range(31))+[114]]))

#%% 
# Dropping the column V14_main_crime, V7_region,
# and V3_nationality_country as they are too specific 
# Also dropping V5_age_cat, as we have the numeric variable 
dfsub.drop(columns=['V2_nationality_type',
                    'V3_nationality_country', 
                    'V5_age_cat', 
                    'V7_region',
                    'V14_main_crime', 
                    'V25_MRM_ATM_or_enforcement_actions'], inplace=True)

#%% Replacing some of the catalan words with english 
V1_map = {'Dona': 'female', 'Home': 'male'}
V4_map = {'Espanya': 'Spain', 
          'Magrib': 'Maghreb', 
          'Centre i Sud Amèrica': 'Latin America',
          'Altres': 'Other', 
          'Europa': 'Europe'}
V11_map = {'Amb antecedents': 1, 'Sense antecedents': 0}
V12_map = {'1 o 2 antecedents': '1-2',
           'De 3 a 5 antecedents': '3-5',
           'Més de 5 antecedents': '5+',
           0: '0'}
V13_map = {'3 fets o més': '3+',
           '2 fets': '2',
           '1 fet' : '1'}
V15_map = {'Contra les persones': 'Against People',
            'Contra la propietat violent': 'Against Property',
            'Contra la propietat no violent': 'Against Property',
            'Altres': 'Other'}
V16_map = {'Violent': 1, 'No violent': 0}
V17_map = {'Delicte': 1, 'Falta': 0}
V26_map = {'Internament': 'Internment',
           'LV': 'Probation', #Llibertat vigilada
           'ATM': 'ATM', #report of technical advice
           'MRM':  'MRM', #meditation
           'Altres medi obert': 'Other', 
           'PBC': 'Community Service'}
V27_map = {'Menys de 6 mesos': '<6 months', 
                'De 6 mesos a 1 any': '6 months < 1 year', 
                "Més d'1 any": '>1 year'}
V115_map = {'Sí': 1, 'No': 0}

dfsub = dfsub.assign(
    V1_sex = lambda x: x.V1_sex.map(V1_map),
    V4_area_origin = lambda x: x.V4_area_origin.map(V4_map),
    V8_age = lambda x: x.V8_age.astype(int),
    V11_criminal_record = lambda x: x.V11_criminal_record.map(V11_map),
    V12_n_criminal_record = lambda x: x.V12_n_criminal_record.map(V12_map),
    V13_n_crime_cat = lambda x: x.V13_n_crime_cat.map(V13_map),
    V15_main_crime_cat = lambda x: x.V15_main_crime_cat.map(V15_map),
    V16_violent_crime = lambda x: x.V16_violent_crime.map(V16_map),
    V17_crime_classification = lambda x: x.V17_crime_classification.map(V17_map),
    V26_finished_measure_grouped = lambda x: x.V26_finished_measure_grouped.map(V26_map),
    V27_program_duration_cat = lambda x: x.V27_program_duration_cat.map(V27_map),
    V115_RECID2015_recid = lambda x: x.V115_RECID2015_recid.map(V115_map)
)

#%% Imputing missing values 

#There are some missing values in 'V28_days_from_crime_to_program
#we impute them by calculating the difference in days from crime committed 
#to start of program
tmp = (dfsub.loc[
    dfsub.V28_days_from_crime_to_program.isnull(),
    ['V22_main_crime_date', 'V30_program_start']
    ])
n_nans = tmp.shape[0]
vals = [(tmp.V22_main_crime_date.iloc[i].date()
        -tmp.V30_program_start.iloc[i].date()).days for i in range(n_nans)]

dfsub.loc[dfsub.V28_days_from_crime_to_program.isnull(), 'V28_days_from_crime_to_program'] = vals 

# Checking for nulls 
#dfsub.isnull().sum(axis=0)

#%% handling dates to extract month and year from each. Day is omitted.
date_cols = dfsub.select_dtypes(include=['datetime64']).columns

for i, col in enumerate(date_cols):
    dfsub[f"{col}_year"] = dfsub[col].dt.year
    dfsub[f"{col}_month"] = dfsub[col].dt.month

dfsub.drop(date_cols, axis = 1, inplace=True)

# 'V31_program_end_year' is 2010 in all instances. Column is dropped.
dfsub.drop(["V31_program_end_year"], axis = 1, inplace=True)

#%% Writing to Excel
dfsub.to_csv(f"{processed_file_path}/catalan-juvenile-recidivism-subset.csv", index = False)