import pandas as pd
raw_file_path = 'data\\raw\\catalan-juvenile-recidivism\\reincidenciaJusticiaMenors.xlsx'
processed_file_path = 'data\\processed\\catalan-juvenile-recidivism'
column_translation_file_path = 'data\\interim\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-columns.xlsx'
    
raw = pd.read_excel(raw_file_path)
columns_trans = pd.read_excel(column_translation_file_path)
raw.head()

# Shape of data
raw.shape # (4753, 143)

# Getting number of Nans in each column: 
raw.isnull().sum(axis = 0)

# Columns with only Nan
raw.columns[(raw.isnull().sum(axis = 0) == raw.shape[0])]

# renaming to english columns and writing file with NaNs 
catalan_names = raw.columns
raw.columns = columns_trans.translation 
(pd.DataFrame(
    raw.isnull().sum(axis= 0)
                .rename('NaNs'))
                .assign(**{'n_no_NaNs': lambda x: raw.shape[0]-x.NaNs, 
                           'catalan_names': catalan_names}) 
                .reindex(columns = ['catalan_names', 'NaNs', 'n_no_NaNs'])
                .reset_index()
                .to_excel("data\\interim\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-NaNs.xlsx")
)

# Dropping columns with only NaN 
df = raw.drop(raw.columns[(raw.isnull().sum(axis = 0) == raw.shape[0])], axis = 1)

# Create dict with unique values for each column 
u_dict = {}
for i, col in enumerate(df.columns): 
    if i >= 18: 
        i = i+1
        
    dict_key = f"V{i}"
    u_dict[dict_key] = df[col].unique()


# Finding nationality of offenders with missing geographical areas of origin 
df.loc[df.V4_area_origin.isnull()].V2_nationality_type.unique()
#Inputting Espanya on the 'missing' values
df.loc[df.V4_area_origin.isnull(), 'V4_area_origin'] = 'Espanya'

# Filling or dropping NaNs 
df.V12_n_criminal_record.fillna(0, inplace = True)
df.V20_n_juvenile_records.fillna(0, inplace = True)
df.V26_finished_measure_grouped.fillna(df.V25_MRM_ATM_or_enforcement_actions, inplace = True)
df.dropna(axis = 0, subset=['V5_age_cat', 'V6_province', 'V7_region', 'V8_age'], inplace = True)

# Subsetting columns 
dfsub = df.filter(list(df.columns[list(range(31))+[114]]))

# Replacing some words with english 
V1_map = {'Dona': 'female', 'Home': 'male'}
V2_map = {'Espanyol': 'Spanish', 'Estranger': 'Foreign'}
V4_map = {'Espanya': 'Spain', 
              'Magrib': 'Maghreb', 
              'Centre i Sud Amèrica': 'Latin America',
              'Altres': 'Other', 
              'Europa': 'Europe'}
V5_map = {'14 i 15 anys': '14-15 years', '16 i 17 anys':'16-17 years'}
V11_map = {'Amb antecedents': 1, 'Sense antecedents': 0}
V12_map = {'1 o 2 antecedents': '1-2',
                'De 3 a 5 antecedents': '3-5',
                'Més de 5 antecedents': '5+',
                '0': '0'}
V13_map = {'3 fets o més': '3+',
           '2 fets': '2',
           '1 fet' : '1'
}
V16_map = {'Violent': 1, 'No violent': 0}
V17_map = {'Delicte': 1, 'Falta': 0}
V27_map = {'Menys de 6 mesos': '<6 months', 
                'De 6 mesos a 1 any': '6 months < 1 year', 
                "Més d'1 any": 'more than 1 year'}
V115_map = {'Sí': 1, 'No': 0}

dfsub = dfsub.assign(
    V1_sex = lambda x: x.V1_sex.map(V1_map),
    V2_nationality_type = lambda x: x.V2_nationality_type.map(V2_map),
    V4_area_origin = lambda x: x.V4_area_origin.map(V4_map),
    V5_age_cat = lambda x: x.V5_age_cat.map(V5_map),
    V8_age = lambda x: x.V8_age.astype(int),
    V11_criminal_record = lambda x: x.V11_criminal_record.map(V11_map),
    V12_n_criminal_record = lambda x: x.V12_n_criminal_record.map(V12_map),
    V13_n_crime_cat = lambda x: x.V13_n_crime_cat.map(V13_map),
    V16_violent_record = lambda x: x.V16_violent_record.map(V16_map),
    V17_crime_classification = lambda x: x.V17_crime_classification.map(V17_map),
    V27_program_duration_cat = lambda x: x.V27_program_duration_cat.map(V27_map),
    V115_RECID2015_recid = lambda x: x.V115_RECID2015_recid.map(V115_map)
)

dfsub.isnull().sum(axis=0)

#There are some missing values in 'V28_days_from_crime_to_program
#we impute them by calculating the difference in days from crime comitted 
#to start of program
tmp = (dfsub.loc[
    dfsub.V28_days_from_crime_to_program.isnull(),
    ['V22_main_crime_comission_date', 'V30_program_start']
    ])
n_nans = tmp.shape[0]
vals = [(tmp.V22_main_crime_comission_date.iloc[i].date()
        -tmp.V30_program_start.iloc[i].date()).days for i in range(n_nans)]

dfsub.loc[dfsub.V28_days_from_crime_to_program.isnull(), 'V28_days_from_crime_to_program'] = vals 

dfsub.to_excel(f"{processed_file_path}\\catalan-juvenile-recidivism-subset.xlsx")