import pandas as pd
raw_file_path = 'data\\raw\\catalan-juvenile-recidivism\\reincidenciaJusticiaMenors.xlsx'
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

# Finding nationality of offenders with missing geographical areas of origin 
df.loc[df.V4_area_origin.isnull()].V2_nationality_type.unique()

#Inputting Espanya on the 'missing' values
df.loc[df.V4_area_origin.isnull(), 'V4_area_origin'] = 'Espanya'

# If V25 == ATM then V24 is 'Assessorament tècnic menors' or 'Assessorament tècnic (art. 4)'and V26 is NaN
# If V25 == MRM then V24 is 'Mediació i reparació menors' and V26 is NaN
# We add a category in V26 for each of these
dfsub = df.loc[(df.V25_MRM_ATM_or_enforcement_actions == "MRM") | (df.V25_MRM_ATM_or_enforcement_actions == "ATM")]
(pd.DataFrame(dfsub.isnull().sum(axis = 0)
              .rename('NaNs'))
              .assign(**{'n_no_NaNs': lambda x: dfsub.shape[0]-x.NaNs})
              .to_excel('data\\interim\\catalan-juvenile-recidivism\\ATM_MRM_Nans.xlsx')
              )

# Create dict with unique values for each column 
u_dict = {}
for i, col in enumerate(df.columns): 
    if i >= 18: 
        i = i+1
        
    dict_key = f"V{i}"
    u_dict[dict_key] = df[col].unique()

# Filling or dropping NaNs 
df.V12_n_criminal_record_cat.fillna(0, inplace = True)
df.V20_n_juvenile_records.fillna(0, inplace = True)
df.V26_finished_measure_grouped.fillna(df.V25_MRM_ATM_or_enforcement_actions, inplace = True)
df.dropna(axis = 0, subset=['V5_age_cat', 'V6_province', 'V7_region', 'V8_age'], inplace = True)

# 
