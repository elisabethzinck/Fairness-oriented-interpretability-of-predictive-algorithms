#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation_tool.layered_tool import FairKit
#%%
if __name__ == '__main__':
    figure_path = 'figures/evaluation_plots/'
    fig_path_report = '../Thesis-report/00_figures/'
    
    run_all = False
    update_report_figures = False # Write new figures to report repository?

    run_anym = False
    run_german = False
    run_taiwanese = True
    run_compas = True
    run_catalan = True
    run_ADNI = False
    adni_no = 2

    credit_w_fp = 0.9
    compas_w_fp = 0.9
    catalan_w_fp = 0.9
    anym_w_fp = 0.2
    adni_w_fp = 0.1

    #######################################
    # Anonymous data (from german)
    #######################################
    if run_anym or run_all:
        file_path = 'data\\processed\\anonymous_data.csv'
        anym = pd.read_csv(file_path)
        fair_anym = FairKit(
            y = anym.y, 
            y_hat = anym.yhat, 
            a = anym.grp, 
            r = anym.phat,
            w_fp = anym_w_fp)
        fair_anym.plot_confusion_matrix()
        plt.savefig(figure_path+'anym_confusion.png')
        fair_anym.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'anym_l2.png')
        if update_report_figures:
            plt.savefig(fig_path_report+'L2_example.pdf', bbox_inches='tight')
        fair_anym.layer_1(output_table=False)



    #######################################
    # German credit logistic regression
    #######################################

    if run_german or run_all:
        file_path = 'data\\predictions\\german_credit_log_reg.csv'
        german_log_reg = pd.read_csv(file_path)

        fair_german_log_reg = FairKit(
            y = german_log_reg.credit_score, 
            y_hat = german_log_reg.log_reg_pred, 
            a = german_log_reg.sex, 
            r = german_log_reg.log_reg_prob,
            w_fp = credit_w_fp,
            model_type='Logistic Regression')
        fair_german_log_reg.plot_confusion_matrix()
        plt.savefig(figure_path+'german_log_reg_confusion.png')
        fair_german_log_reg.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'german_log_reg_l2.png')
        fair_german_log_reg.layer_1(output_table=False)
        

    #######################################
    # German credit neural network
    #######################################
    if run_german or run_all:
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
            w_fp = credit_w_fp,
            model_type='Neural network')
        fair_german_nn.plot_confusion_matrix()
        plt.savefig(figure_path+'german_nn_confusion.png')
        fair_german_nn.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'german_nn_l2.png')
        fair_german_nn.layer_1(output_table=False)
        


    #######################################
    # Compas decile score
    #######################################
    if run_compas or run_all:
        compas_file_path = 'data\\processed\\compas\\compas-scores-two-years-pred.csv'
        compas = pd.read_csv(compas_file_path)

        fair_compas_age = FairKit(
            y = compas.two_year_recid, 
            y_hat = compas.pred_medium_high, 
            a = compas.age_cat, 
            r = compas.decile_score,
            w_fp = compas_w_fp,
            model_type='COMPAS Decile Scores')
        fair_compas_age.plot_confusion_matrix()
        plt.savefig(figure_path+'compas_confusion_age.png')
        fair_compas_age.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'compas_l2_age.png')
        fair_compas_age.layer_1(output_table=False)

        # filtering out hispanics to recreate the Propublica result 
        not_include_hispanics = False
        if not_include_hispanics: 
            compas = compas[compas.race.isin(['African-American','Caucasian'])]

        fair_compas_race = FairKit(
            y = compas.two_year_recid, 
            y_hat = compas.pred_medium_high, 
            a = compas.race, 
            r = compas.decile_score,
            w_fp = compas_w_fp,
            model_type='COMPAS Decile Scores')
        fair_compas_race.plot_confusion_matrix()
        plt.savefig(figure_path+'compas_confusion_race.png')
        fair_compas_race.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'compas_l2_race.png')
        fair_compas_race.layer_1(output_table=False)
    

    #######################################
    # Catalan Juvenile Recidivism NN
    #######################################
    if run_catalan or run_all:
        catalan_file_path = 'data/predictions/catalan-juvenile-recidivism/catalan_recid_nn_pred.csv'
        catalan = pd.read_csv(catalan_file_path)

    # Sensitive: Area of Origin
        fair_catalan = FairKit(
            y = catalan.V115_RECID2015_recid, 
            y_hat = catalan.nn_pred, 
            a = catalan.V4_area_origin, 
            r = catalan.nn_prob,
            w_fp = catalan_w_fp,
            model_type='Catalan NN')

        fair_catalan.plot_confusion_matrix()
        plt.savefig(figure_path+'catalan_confusion_V4_area_origin.png')
        fair_catalan.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'catalan_l2_V4_area_origin.png')
        fair_catalan.layer_1(output_table=False)


    #######################################
    # Taiwanese NN
    #######################################
    if run_taiwanese or run_all:
        taiwanese_file_path = 'data/predictions/taiwanese_nn_pred.csv'
        taiwanese = pd.read_csv(taiwanese_file_path)

    # Sensitive: Nationality Type 
        fair_taiwanese = FairKit(
            y = taiwanese.default_next_month, 
            y_hat = taiwanese.nn_pred, 
            a = taiwanese.sex, 
            r = taiwanese.nn_prob,
            w_fp = credit_w_fp,
            model_type='Taiwanese NN')

        fair_taiwanese.plot_confusion_matrix()
        plt.savefig(figure_path+'taiwanese_confusion_sex.png')
        fair_taiwanese.layer_2(plot = True, output_table=False)
        plt.savefig(figure_path+'taiwanese_l2_sex.png')
        fair_taiwanese.layer_1(output_table=False)

    if run_ADNI or run_all:
        for adni_no in [1,2]:
            file_path = f'data/ADNI/predictions/ADNI_{adni_no}_nn_pred.csv'
            adni = pd.read_csv(file_path)
            fair_adni = FairKit(
                y = adni.y, 
                y_hat = adni.nn_pred, 
                a = adni.sex, 
                r = adni.nn_prob,
                w_fp = adni_w_fp,
                model_type=f'ADNI{adni_no} NN')
            fair_adni.plot_confusion_matrix()
            plt.savefig(figure_path+f'adni{adni_no}_confusion.png')
            fair_adni.layer_2(plot = True, output_table=False)
            plt.savefig(figure_path+f'adni{adni_no}_l2.png')
            fair_adni.layer_1(output_table=False)
            
            


# %%
