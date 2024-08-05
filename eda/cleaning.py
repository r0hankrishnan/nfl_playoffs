#This file was used to create a dataset ready for predictive modelling

#Packages
import pandas as pd

#Load data
path = "../../nfl-playoff-predictor/data/nfl_data.xlsx"
df = pd.read_excel(path)

#Look at first columns
df.head()

#Look at desciptive stats
df.describe()