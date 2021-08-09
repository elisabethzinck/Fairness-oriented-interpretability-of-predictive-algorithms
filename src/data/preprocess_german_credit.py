#%% Initialization
import pandas as pd
input_path = '..\\..\\data\\raw\\german_credit\\german.data'
output_path = '..\\..\\data\\processed\\german_credit.csv'
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



#%% Cleaning personal_status_sex
df.personal_status_sex.unique()

# Create sex column (from german_credit.doc)
sex_map = {
    'A91': 'male',
    'A92': 'female',
    'A93': 'male',
    'A94': 'male', 
    'A95': 'female'
    }
df['sex'] = df['personal_status_sex'].map(sex_map)

#%% Removing single males and female (because no female singles)
df = df[
    (df.personal_status_sex != 'A93') & \
    (df.personal_status_sex != 'A95')]


#%% Writing processed data
df.to_csv(output_path, index = False)
# %%
