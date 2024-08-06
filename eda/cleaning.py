#This file was used to create a dataset ready for predictive modelling

#Packages
import pandas as pd


#Load data
path = "../../nfl/data/nfl_data.xlsx"
df = pd.read_excel(path)
year = df["Year"]

#Look at first columns
df.head()

#Look at desciptive stats
df.describe()

#Only want per game stats
#Drop Team, total stats, and Year since those can't be used as predictors
drop = list(["Team","off_gp","off_total_yds","off_pass_yds","off_rush_yds","off_pts","def_total_yds"
             ,"def_pass_yds","def_rush_yds","def_pts","Year"])
dfModel = df.drop(drop, axis = 1)

dfModel.head()

#Save modelling data set to data folder
dfModel.to_csv("../../nfl/data/modelling.csv")