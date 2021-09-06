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

# Dropping columns with only NaN 
df = raw.drop(raw.columns[(raw.isnull().sum(axis = 0) == raw.shape[0])], axis = 1)

(raw.isnull().sum(axis= 0)
             .sort_values(ascending=False)
             .rename('NaNs')
             .to_excel("data\\interim\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-NaNs.xlsx")
)

(pd.DataFrame(
    raw.isnull().sum(axis= 0)
                .rename('NaNs'))
                .assign(**{'N_no_NaNs': lambda x: raw.shape[0]-x.NaNs})
                .to_excel("data\\interim\\catalan-juvenile-recidivism\\catalan-juvenile-recidivism-NaNs.xlsx")
)