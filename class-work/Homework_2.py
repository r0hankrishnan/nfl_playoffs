#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:26:15 2023

@author: rohankrishnan
"""
#ln[3]:
# # Notebook Table of Contents
# ## 1. Exploratory Data Analysis and Data Cleaning 
# 
# #### EDA: Load Data/Variable Summary Statistics, Summary Statistics by Quality group, Quality value counts, Correlation Matrix,Plotting distributions, Plotting distributions by quality.
# #### Data Cleaning: Removing Missing Variables, Removing outliers which were due to manual entry error.
# 
# 
# ## 2. Modeling: Binary Classification
# 
# #### Binarized response to 0 or 1 indicating low vs high quality. Values of 7 or 8 are denoted as high quality and those below are denoted as low quality.Threshold of above 6 chosen as quality of 6 represents 75 percentile in quality.
# #### Models employed were Gaussian Naive Bayes and kNN. To select k value for kNN, used 5 fold cross validation, and iteratively constructed models with different k values (1 - 40)range of 1 - 40 is chosen because our data set is roughly 1600 observations.
# #### When creating this model I employed a pipeline for feature tuning using the minmax scaler, this scaled the data to a smaller range. I then assessed model accuracy for each k value and found the best k-value with the highest accuracy measured by classification rate. Then used the best k value to calculate the average accuracy over 5 fold cross validation.
# #### Determined final model by comparing accuracy of gnb and kNN
#     
# ## 3. Prediction Function 
# #### Created prediction function which allows sommelier to input wine characteristics and then it indicates if the wine is high or low quality.
#

### Importing basic packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

### Load in NFL data
path = "/Users/rohankrishnan/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Fall 2023/ECON 1700/hw2_nfl_final_data.xlsx"
nfl = pd.read_excel(path)

#Head function to see all variables
nfl.head()

#Move response to front of dataset
cols = list(nfl.columns.values);cols

nfl = nfl[[ 'Playoff',
 'Team',
 'off_gp',
 'off_total_yds',
 'off_total_yds_g',
 'off_pass_yds',
 'off_pass_yds_g',
 'off_rush_yds',
 'off_rush_yds_g',
 'off_pts',
 'off_pts_g',
 'def_total_yds',
 'def_total_yds_g',
 'def_pass_yds',
 'def_pass_yds_g',
 'def_rush_yds',
 'def_rush_yds_g',
 'def_pts',
 'def_pts_g',
 'kick_att',
 'kick_yds',
 'kick_avg',
 'kick_lng',
 'kick_td',
 'punt_att',
 'punt_yds',
 'punt_avg',
 'punt_lng',
 'punt_td',
 'punt_fc',
 'turn_diff',
 'take_int',
 'take_fum',
 'take_total',
 'give_int',
 'give_fum',
 'give_total',
 'record',
 'rat',
 'pwr',
 'off',
 'def',
 'hfa',
 'sos',
 'Year',]]

#Describe function to see distributions of the variables
nfl.describe()

#Make sure there are no NA values
nfl.isna().sum()

#Check which variables are continuous and which are discrete
nfl.nunique()

#Import analysis packages
import seaborn as sns
#pip install joypy
import joypy

#Slightly more No than Yes-- makes sense since only 12 (and later 14) teams make it
nfl["Playoff"].value_counts()

#Conditional means
cond_means = nfl.groupby("Playoff").mean()

#   ###Takeaways from conditional means
#   off_gp: No real difference-- makes sense and will probably remove this
#   offensive stats: 1 had a higher mean than 0 for all stats
#   defensive stats: Much more even. There were a couple where 0 had a higher 
#   mean than 1 but none the other way
#   speical stats: More mixed, 1 seemed to take less kicks but were longer and 
#   scored more often --> similar trend with punts
#   Rest were as expected: 1 averaging higher on offensive stats

#Create correlation matrix
correlation_matrix = nfl.corr()

### Create a heatmap of the correlation matrix
plt.figure(figsize=(25, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# ###Correlation matrix takeaways:
#    Strongest Positive Correlations: record, rat, pwr, off, def, off_pts
#    Strongest Negative Correlations: def_pts, def_pts_g  
#    Potential Collinearity Issues: def Massey rating and defensive ESPN stats, 
#    pwr and off Massey ratings and offensive ESPN stats

#Density plot matrix with labels LFGGGG
features = nfl.columns[2:44]
nfl_features = nfl[features]
nfl_f_melt= nfl_features.melt(var_name = "cols", value_name = "vals")
sns.displot(kind='kde', data=nfl_f_melt, col='cols', col_wrap=8, x='vals',
            hue="cols", facet_kws={'sharey': False, 'sharex': False})

#Individual density plots to look at specific distributions
for i in nfl_features.columns:
    plt.figure()
    sns.kdeplot(nfl_features[i])

#Select columns where at least one row is negative
nfl_features.loc[:, (nfl_features<0).any()]
#turn_diff, pwr, def, and sos have negative values that indicate real information

#Transform turn_diff, pwr, def, and sos to positive numbers by adjusting their min value to be 1.001
nfl_negative = nfl_features.copy()
nfl_negative["turn_diff"] = np.log(nfl_negative["turn_diff"] + np.abs(np.min(nfl_negative["turn_diff"])) + 1.001)
nfl_negative["pwr"] = np.log(nfl_negative["pwr"] + np.abs(np.min(nfl_negative["pwr"])) + 1.001)
nfl_negative["def"] = np.log(nfl_negative["def"]+ np.abs(np.min(nfl_negative["def"])) + 1.001)
nfl_negative["sos"] = np.log(nfl_negative["sos"]+ np.abs(np.min(nfl_negative["sos"])) + 1.001)

nfl_negative.loc[:, (nfl_negative<0).any()] #No more negative values

#Create model dataset
nfl_model = nfl.copy()

#Drop games, Team, Year, and total variables (choosing to use per game stats rather than total)
drop = list(["Team","off_gp","off_total_yds","off_pass_yds","off_rush_yds","off_pts","def_total_yds"
             ,"def_pass_yds","def_rush_yds","def_pts","Year"])
nfl_model = nfl_model.drop(drop, axis = 1)

#Add in transformed turn_diff, pwr, def, and sos
nfl_model["t_turn_diff"] = nfl_negative["turn_diff"]
nfl_model["t_pwr"] = nfl_negative["pwr"]
nfl_model["t_def"] = nfl_negative["def"]
nfl_model["t_sos"] = nfl_negative["sos"]

#Drop turn_diff, pwr, def, and sos
nfl_model = nfl_model.drop(["turn_diff", "pwr", "def", "sos"], axis=1)

#Create team and year indices
Team = list(nfl["Team"])
Year = list(nfl["Year"])

#   ### Models Developed 
# I will start with a Naive Bayes classifier followed by a probit regression and a bagged tree

#   ### Naive Bayes
# machine learning library
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

# library to work with data
import pandas as pd

# libraries to plot confusion matrices
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#Create X, y, and individual training/testing sets
X = nfl_model.drop("Playoff", axis = 1)
y = nfl_model["Playoff"]

X["Year"] = Year
y = pd.DataFrame(y)
y["Year"] = Year

#Filter by before and after 2022 for train and test sets and remove Year variable
X_train = X.loc[X["Year"] < 2022]
y_train = y.loc[y["Year"]<2022]
X_test = X.loc[X["Year"]==2022]
y_test = y.loc[y["Year"]==2022]
X_train = X_train.drop("Year", axis = 1)
X_test = X_test.drop("Year", axis = 1)
y_train = y_train.drop("Year",axis = 1)
y_test = y_test.drop("Year", axis = 1)

X = X.drop("Year",axis = 1)
y = y.drop("Year", axis = 1)

#Define the Naive Bayes Classifier
nb_model = BernoulliNB()

# Train the model 
nb_model.fit(X_train, y_train)

# Predict Output using the test data
pred = nb_model.predict(X_test)

### import the cross_val score to be used 
from sklearn.model_selection import cross_val_score

### calculate cross validation scores across different folds
scores = cross_val_score(nb_model, X, y, cv = 10, scoring='accuracy')

### print scores for each fold
print('Cross-validation scores:{}'.format(scores))

### We now calculate the the average score using the mean 
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
### Average accuracy is 0.6042 -- we can defnitely do better

#   ###Linear Regression
from sklearn.utils import shuffle
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd

probit = sm.Probit(y_train, X_train)
results_probit = probit.fit()

pred_probit = results_probit.predict(X_test)
pred_probit = pd.DataFrame(data = pred_probit, columns=["y_pred"])
probit_df= pred_probit.copy()
probit_df["y_test"] = y_test

print(probit_df['y_test'].describe())
print(probit_df["y_pred"].describe())

pred_probit = np.array(pred_probit)

pred_probit_thres = np.where(pred_probit > 0.5, 1, 0)

probit_confusion_matrix = confusion_matrix(y_test, pred_probit_thres)
#Accuracy is 0.6667 -- a little better than NB

#   ###Classification Tree
#Libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Define the basic tree classifier
tree = DecisionTreeClassifier(random_state = 0)

# fit it to the training data
tree.fit(X_train, y_train)
tree.score(X_test,y_test)

# compute accuracy in the test data
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
#Accuracy is 0.750 -- even better than the other two

#Plot tree
plt.figure("Decision Tree", figsize=[20,20])
plot_tree(tree, fontsize=(15), filled = True, feature_names = X.columns, class_names= ["Yes", "No"])
plt.show()
    
plot_confusion_matrix(tree, X_test, y_test, display_labels=["No", "Yes"])


#Visualize alpha
comp_path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = comp_path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

tree_dts = []

for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state = 0, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    tree_dts.append(tree)
    
train_scores = [tree.score(X_train, y_train)for tree in tree_dts]
test_scores = [tree.score(X_test, y_test)for tree in tree_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("alpha vs accuracy for training vs test")
ax.plot(ccp_alphas, train_scores, marker = "o", label = "train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = "o", label = "test", drawstyle = "steps-post")
ax.legend()
plt.show()
#Around 0.012 seems to be ideal


#Create tree with new ccp
final_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)

final_tree.fit(X_train, y_train)
final_tree.score(X_test,y_test)

print("Accuracy on test set: {:.3f}".format(final_tree.score(X_test, y_test)))
#Accuracy is 0.844, best yet!

#   ###Use final_tree to predict on 2023 data
#Load 2023 data
path_2023 = "/Users/rohankrishnan/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Fall 2023/ECON 1700/2023_pred_data.xlsx"
nfl_2023 = pd.read_excel(path_2023)

#Create list of features used in tree
X_cols = list(X.columns.values);X_cols

#Create new dataset to transform negative values
nfl_2023_neg = nfl_2023.copy()

#Create lists for teams and year since they will be removed
teams_2023 = nfl_2023_neg["Team"]
year_2023 = nfl_2023_neg["Year"]

#Remove "Team" and "Year"
nfl_2023_neg = nfl_2023_neg.drop(["Team", "Year"],axis =1)

#Check negative values
nfl_2023_neg.loc[:, (nfl_2023_neg<0).any()]

#Convert negative values to positive
nfl_2023_neg["turn_diff"] = np.log(nfl_2023_neg["turn_diff"] + np.abs(np.min(nfl_2023_neg["turn_diff"])) + 1.001)
nfl_2023_neg["pwr"] = np.log(nfl_2023_neg["pwr"] + np.abs(np.min(nfl_2023_neg["pwr"])) + 1.001)
nfl_2023_neg["def"] = np.log(nfl_2023_neg["def"]+ np.abs(np.min(nfl_2023_neg["def"])) + 1.001)
nfl_2023_neg["sos"] = np.log(nfl_2023_neg["sos"]+ np.abs(np.min(nfl_2023_neg["sos"])) + 1.001)

#Check negative values
nfl_2023_neg.loc[:, (nfl_2023_neg<0).any()] #Got rid of negative values

#Create test set
nfl_2023_model = nfl_2023.drop(["Team","Year"],axis =1).copy()


#Add in transformed turn_diff, pwr, def, and sos
nfl_2023_model["t_turn_diff"] = nfl_2023_neg["turn_diff"]
nfl_2023_model["t_pwr"] = nfl_2023_neg["pwr"]
nfl_2023_model["t_def"] = nfl_2023_neg["def"]
nfl_2023_model["t_sos"] = nfl_2023_neg["sos"]

#Filter test set so it matches X
nfl_2023_model = nfl_2023_model[X_cols]

#Call final_tree
final_tree = DecisionTreeClassifier(random_state=0, ccp_alpha = 0.0012)

final_tree.fit(X_train, y_train)

#See variable importance-- Record is by far the most important predictor-- makes sense!
f_imp = final_tree.feature_importances_
f_imp = pd.DataFrame(f_imp)
f_imp["var"] = X_cols

#Get final predictions
pred_2023 = final_tree.predict(nfl_2023_model)
pred_2023 = pd.DataFrame(pred_2023)
pred_2023["Team"] = teams_2023






