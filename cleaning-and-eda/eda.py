#Load data manipulation packages
import pandas as pd

#-------------------------------#
path = "../../nfl-playoff-predictor/data/nfl_data.xlsx"
df = pd.read_excel(path)

#Look at first columns
df.head()

#Look at desciptive stats
df.describe()

#Remove team values
team = df["Team"]
dfNoTeam = df.drop("Team", axis = 1).copy()


#Check for NA values
numNA = 0
for col in df.columns:
    if df[col].isna().sum() == 0:
        numNA += 1
        pass
    else:
        print(f"{col} has {df[col].isna().sum()} NA values.")

print(f"There are {numNA} columns with NA values")

#List unique values -- look for potential discrete variables
for col in df.columns:
    if df[col].nunique() <= 10:
        print(f"{col} has {df[col].nunique()} unique values.")
    else:
        pass
    
#off_gp, kick_td, punt_td, Year, and Playoff have under 10 unique values
for col in ["off_gp", "kick_td", "punt_td", "Year", "Playoff"]:
    print(f"{col} has the following values: {df[col].unique()}")
    
#-------------------------------#

# Look at proportion of teams making playoffs vs not
df["Playoff"].value_counts() #174 NO vs 114 YES -- approx. 60% of obs are 0

#Compare means between playoff and non playoff teams
conditionalMeans = df.groupby("Playoff").mean(numeric_only= True)

for col in conditionalMeans.columns: 
    print(f"{col} Playoff - No Playoff: {conditionalMeans[col][1]-conditionalMeans[col][0]}")
    
#Create df of differences so we can sort them    
conditionalDifferences = pd.DataFrame({"var":[], "diff":[]})

ind = 0
for col in conditionalMeans.columns:
    diff = conditionalMeans[col][1]-conditionalMeans[col][0]
    new_row = pd.DataFrame({"var": col, "diff": diff}, index = [ind])
    conditionalDifferences = pd.concat([conditionalDifferences, new_row])
    ind = ind + 1

conditionalDifferences.sort_values(by = "diff")
    
#Want to see which var has the largest difference between groups:
#First, standardized columns:
def standardize(column):
    return (column - column.mean()) / column.std()

#Create standardized df
dfStandard = df.copy()
Playoff = dfStandard["Playoff"]
dfStandard = dfStandard.drop(["Team", "Playoff"], axis = 1)
dfStandard = dfStandard.apply(standardize)
dfStandard["Playoff"] = Playoff

#Check if it worked
for col in dfStandard.columns:
    print(f"{col} mean: {round(dfStandard[col].mean(), 5)}")
    
del(Playoff)

condStandMeans = dfStandard.groupby("Playoff").mean(numeric_only=True)

condStandDiffs = pd.DataFrame({"var":[], "diff":[]})

ind = 0
for col in condStandMeans:
    diff = condStandMeans[col][1] - condStandMeans[col][0]
    new_row = pd.DataFrame({"var": col, "diff": diff}, index= [ind])
    condStandDiffs = pd.concat([condStandDiffs, new_row])
    ind = ind + 1
    
condStandDiffs.sort_values(by= "diff")

#-------------------------------#

#Load visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

#Individual density plots to look at specific distributions
for i in df.copy().drop(["Team", "Playoff"], axis = 1).columns:
    plt.figure()
    sns.kdeplot(df[i])
    
#Scatter plot matrix -- unreadable 
pd.plotting.scatter_matrix(df.copy().drop(["Team", "Playoff"], axis = 1))


#Correlation matrix
corr_mat = df.drop("Team", axis  = 1).corr()
plt.figure(figsize =(25, 20))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


