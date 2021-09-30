# imports 
import pandas as pd 
import numpy as np 
import os

from src.visualization_description.descriptive_functions import DescribeData

# reading data 
df = pd.read_csv("data/processed/compas/compas-scores-two-years-pred.csv")

desc = DescribeData(data = df, 
                    y_name = "two_year_recid",
                    a_name = 'race', 
                    id_name = 'id')

desc.agg_table(to_latex=False, target_tex_name="Recidivists")
