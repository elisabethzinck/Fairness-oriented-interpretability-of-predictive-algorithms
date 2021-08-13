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
    data = pd.read_csv(file_path)
    data.head()
    n = data.shape[0]
    
    X = data[data.columns.difference(['credit_score'])]
    X = one_hot_encode_mixed_data(X)

    # Generate random classifications
    random.seed(60220)
    preds = random.choices([0,1], k = n)
    data['prediction'] = preds

    # Making logistic regression 
    log_reg = LogisticRegression(penalty='none')
    log_reg.fit(X, data.credit_score)
    data["logistic_regression_prediction"] = log_reg.predict(X)

    fair_log_reg = EvaluationTool(
        y = data.credit_score, 
        c = data.logistic_regression_prediction, 
        a = data.sex,
        tol = 0.01,
        model_type='Logistic Regression')
    fair_log_reg.get_confusion_matrix()
    fair_log_reg.plot_confusion_matrix()
    fair_log_reg.get_rates()
    fair_log_reg.separation()

    fair_rand = EvaluationTool(
        y = data.credit_score, 
        c = data.prediction, 
        a = data.sex, 
        model_type='Random')
    fair_rand.get_confusion_matrix()
    fair_rand.plot_confusion_matrix()
