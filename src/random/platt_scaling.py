# Function for platt scaling
import pandas as pd 
from src.evaluation_tool.basic_tool import EvaluationTool
if __name__ == "__main__":
    file_path = 'data\\processed\\german_credit_pred.csv'
    data = pd.read_csv(file_path)
    data.head()

    fair = EvaluationTool(
        y = data.credit_score, 
        c = data.log_reg_pred, 
        a = data.sex, 
        model_type='Logistic Regression')

    target = data.log_reg_pred
    N_plus = sum(target == 1)
    N_minus = sum(target == 0)
    
    # TODO: Mutate target to get transformed
    #target_scaled