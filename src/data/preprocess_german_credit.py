# Preprocess the german credit data to make it easier to work with
#%% Initialization
import pandas as pd
input_path = 'data\\raw\\german_credit\\german.data'
output_path = 'data\\processed\\german_credit.csv'

# The data was downloaded from 
# https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
# Link from Fairness definitions explained article

#%% Loading data
attribute_names = [
    "account_amount",
    "duration",
    "credit_history",
    "credit_purpose",
    "credit_amount",
    "savings_amount",
    "employment_length",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_length", 
    "property",
    "age",
    "installment_plans",
    "housing",
    "existing_credits", 
    "job",
    "dependents",
    "telephone", 
    "foreign_worker",
    "credit_score"
]

raw_data = pd.read_csv(
    input_path, 
    sep = ' ', 
    names = attribute_names)

df = raw_data.copy()

#%% Create id column to allow for saving predictions without remaining data
df['person_id'] = range(df.shape[0])


#%% Cleaning personal_status_sex
df.personal_status_sex.unique()

# Create sex column (from german_credit.doc)
sex_map = {
    'A91': 'Male',
    'A92': 'Female',
    'A93': 'Male',
    'A94': 'Male', 
    'A95': 'Female'
    }
personal_status_map = {
    'A91': 'married/previously_married',
    'A92': 'married/previously_married',
    'A93': 'single',
    'A94': 'married/previously_married', 
    'A95': 'single'
}
df['sex'] = df['personal_status_sex'].map(sex_map)
df['personal_status'] = df['personal_status_sex'].map(personal_status_map)
df = df.drop(columns = ['personal_status_sex'])



#%% making target binary 
# Original: 1 = Good, 2 = Bad
# New: 0 = Good, 1 = Bad (the same as taiwanese)
df['credit_score'] = df.credit_score - 1

#%% Writing processed data
df.to_csv(output_path, index = False)
# %%
