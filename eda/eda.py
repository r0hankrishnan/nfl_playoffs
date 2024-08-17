#Load data manipulation packages
import pandas as pd

#-------------------------------#
path = "../../nfl/data/nfl_data.xlsx"
df = pd.read_excel(path)

#Look at first columns
df.head()

#Look at desciptive stats
df.describe()

#Remove team values
team = df["Team"]
dfNoTeam = df.drop("Team", axis = 1).copy()


#Check for NA values
noNumNA = 0
for col in df.columns:
    if df[col].isna().sum() == 0:
        noNumNA += 1
        pass
    else:
        print(f"{col} has {df[col].isna().sum()} NA values.")

print(f"There are {noNumNA} columns with no NA values")

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

#Look at table of correlations
correlations = dfNoTeam.corr()
correlations = correlations.loc[:, "Playoff"]
correlations = correlations.reset_index().rename(columns = {"index": "variable"})
correlations = correlations[(correlations["Playoff"] > 0.50) | (correlations["Playoff"] < -0.50)]
correlations.sort_values(by = "Playoff")
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


#Playoffs vs not stat comparison
conditionalMeans = pd.DataFrame(conditionalMeans.to_records()).reset_index()
conditionalMeansMelted = pd.melt(conditionalMeans, id_vars= ["Playoff"], 
                                 value_vars=["rat", "pwr", "def", "hfa", "sos"])

plot = sns.catplot(
    data = conditionalMeansMelted, kind = "bar",
    x = "value", y = "variable", hue = "Playoff",
    errorbar = "sd", palette = "dark", alpha = .6, height = 6
)
plot.despine(left = True)
plot.set_xlabels("Value")
plot.set_ylabels("Massey Stat")
plot.set_titles("")

#Plotly implementation
import plotly.express as px
conditionalMeansMelted["Playoff"] = conditionalMeansMelted["Playoff"].astype(str)
conditionalMeansMelted["Playoff"] = pd.Series(np.where(conditionalMeansMelted["Playoff"].values == '1', "Yes", "No"),
          conditionalMeansMelted.index)

fig = px.bar(conditionalMeansMelted, x="value", y="variable", color="Playoff", barmode="group",
             labels={
                 "value":"Value",
                 "variable":"Massey Stats"
             },
             title="Massey Stats")
fig.update_layout(legend_traceorder="reversed")
fig.show()
