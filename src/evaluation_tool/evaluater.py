#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit
#%%
if __name__ == '__main__':
    figure_path = 'figures/evaluation_plots/'
    fig_path_report = '../Thesis-report/00_figures/'
    
    run_anym = True

    run_german_log_reg = True
    run_german_nn = True

    run_compas = True
    run_catalan = True

    update_report_figures = False # Write new figures to report repository?

    credit_w_fp = 0.9

    compas_w_fp = 0.9
    anym_w_fp = 0.2

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
        plt.savefig(figure_path+'anym_confusion.png')
        fair_anym.l2_plot(w_fp = anym_w_fp)
        plt.savefig(figure_path+'anym_l2.png')
        if update_report_figures:
            plt.savefig(fig_path_report+'L2_example.pdf', bbox_inches='tight')




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
        plt.savefig(figure_path+'german_log_reg_confusion.png')
        fair_german_log_reg.l2_plot(w_fp=credit_w_fp)
        plt.savefig(figure_path+'german_log_reg_l2.png')
        

    #######################################
    # German credit neural network
    #######################################
    if run_german_nn:
        remove_singles = False
        file_path = 'data\\predictions\\german_credit_nn_pred.csv'
        german_nn = pd.read_csv(file_path)
        
        # Remove single males
        if remove_singles:
            german_orig = pd.read_csv(
                'data\\processed\\german_credit_full.csv', 
                usecols=['person_id', 'personal_status'])
            german_nn = (pd.merge(german_orig, german_nn)
                .query('personal_status != "single"'))

        fair_german_nn = FairKit(
            y = german_nn.credit_score, 
            y_hat = german_nn.nn_pred, 
            a = german_nn.sex, 
            r = german_nn.nn_prob,
            model_type='Neural network')
        fair_german_nn.plot_confusion_matrix()
        plt.savefig(figure_path+'german_nn_confusion.png')
        fair_german_nn.l2_plot(w_fp=credit_w_fp)
        plt.savefig(figure_path+'german_nn_l2.png')


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
        plt.savefig(figure_path+'compas_confusion_age.png')
        fair_compas_age.l2_plot(w_fp=compas_w_fp)
        plt.savefig(figure_path+'compas_l2_age.png')

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
        plt.savefig(figure_path+'compas_confusion_race.png')
        fair_compas_race.l2_plot(w_fp=compas_w_fp)
        plt.savefig(figure_path+'compas_l2_race.png')
    
    if run_catalan:
        catalan_file_path = 'data\\predictions\\catalan-juvenile-recidivism\\catalan_recid_nn_pred.csv'
        catalan = pd.read_csv(catalan_file_path)


        fair_catalan = FairKit(
            y = catalan.V115_RECID2015_recid, 
            y_hat = catalan.nn_pred, 
            a = catalan.V4_area_origin, 
            r = catalan.nn_prob,
            model_type='Catalan NN')

        fair_catalan.plot_confusion_matrix()
        fair_catalan.l2_plot(w_fp=0.9)

# %%
