#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.evaluation_tool.basic_tool import EvaluationTool 
from src.data.general_preprocess_functions import one_hot_encode_mixed_data
import random
import os
#%%

if __name__ == "__main__":
    print(os.getcwd())
    file_path = 'data\\processed\\german_credit.csv'
    output_path = 'data\\processed\\german_credit_pred.csv'
    data = pd.read_csv(file_path)
    data.head()
    n = data.shape[0]
    
    X = data[data.columns.difference(['credit_score'])]
    X = one_hot_encode_mixed_data(X)

    # Making logistic regression 
    # To do: Research why logreg gives Convergence Warning
    log_reg = LogisticRegression(penalty='none')
    log_reg.fit(X, data.credit_score)
    data["logistic_regression_prediction"] = log_reg.predict(X)

    data.to_csv(output_path, index = False)

# %%
