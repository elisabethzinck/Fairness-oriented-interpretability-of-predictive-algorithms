#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit

if __name__ == '__main__':
    figure_path = 'figures\\evaluation_plots\\'
    run_anym = False
    run_german_log_reg = False
    run_german_nn = False
    run_compas = True

    german_w_fp = 0.2
    compas_w_fp = 0.9

    #######################################
    # Anonymous data (from german)
    #######################################
    if run_anym:
        file_path = 'data\\processed\\anonymous_data.csv'
        anym = pd.read_csv(file_path)
        fair_anym = FairKit(
            y = anym.y, 
            y_hat = anym.yhat, 
            a = anym.grp, 
            r = anym.phat)
        fair_anym.l1_get_data()
        fair_anym.plot_confusion_matrix()
        plt.savefig(figure_path+'anym_confusion.pdf')
        fair_anym.l2_plot(w_fp=0.2)
        plt.savefig(figure_path+'anym_l2.pdf')
        #plt.savefig('../Thesis-report/00_figures/L2_example_new.pdf', bbox_inches='tight')

    #######################################
    # German credit logistic regression
    #######################################

    if run_german_log_reg:
        file_path = 'data\\predictions\\german_credit_log_reg.csv'
        german_log_reg = pd.read_csv(file_path)

        fair_german_log_reg = FairKit(
            y = german_log_reg.credit_score, 
            y_hat = german_log_reg.log_reg_pred, 
            a = german_log_reg.sex, 
            r = german_log_reg.log_reg_prob,
            model_type='Logistic Regression')
        fair_german_log_reg.plot_confusion_matrix()
        plt.savefig(figure_path+'german_log_reg_confusion.pdf')
        fair_german_log_reg.l2_plot(w_fp=german_w_fp)
        plt.savefig(figure_path+'german_log_reg_l2.pdf')
        

    #######################################
    # German credit neural network
    #######################################
    if run_german_nn:
        file_path = 'data\\predictions\\german_credit_nn_pred.csv'
        german_nn_raw = pd.read_csv(file_path)
        
        # Remove single males
        german_orig = pd.read_csv(
            'data\\processed\\german_credit_full.csv', 
            usecols=['person_id', 'personal_status_sex'])
        german_nn = pd.merge(german_orig, german_nn_raw).query('personal_status_sex != "A93"')

        fair_german_nn = FairKit(
            y = german_nn.credit_score, 
            y_hat = german_nn.nn_pred, 
            a = german_nn.sex, 
            r = german_nn.nn_prob,
            model_type='Neural network')
        fair_german_nn.plot_confusion_matrix()
        plt.savefig(figure_path+'german_log_reg_confusion.pdf')
        fair_german_nn.l2_plot(w_fp=german_w_fp)
        plt.savefig(figure_path+'german_log_reg_l2.pdf')


    #######################################
    # Compas decile score
    #######################################
    if run_compas:
        compas_file_path = 'data\\processed\\compas\\compas-scores-two-years-pred.csv'
        compas = pd.read_csv(compas_file_path)

        fair_compas_age = FairKit(
            y = compas.two_year_recid, 
            y_hat = compas.pred_medium_high, 
            a = compas.age_cat, 
            r = compas.decile_score,
            model_type='COMPAS Decile Scores')
        fair_compas_age.plot_confusion_matrix()
        plt.savefig(figure_path+'compas_confusion_age.pdf')
        fair_compas_age.l2_plot(w_fp=compas_w_fp)
        plt.savefig(figure_path+'compas_l2_age.pdf')

        # filtering out hispanics to recreate the Propublica result 
        not_include_hispanics = False
        if not_include_hispanics: 
            compas = compas[compas.race.isin(['African-American','Caucasian'])]

        fair_compas_race = FairKit(
            y = compas.two_year_recid, 
            y_hat = compas.pred_medium_high, 
            a = compas.race, 
            r = compas.decile_score,
            model_type='COMPAS Decile Scores')
        fair_compas_race.plot_confusion_matrix()
        plt.savefig(figure_path+'compas_confusion_race.pdf')
        fair_compas_race.l2_plot(w_fp=compas_w_fp)
        plt.savefig(figure_path+'compas_l2_race.pdf')

    
# %%
