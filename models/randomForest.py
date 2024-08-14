#Import basic modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
path = "../../nfl/data/modelling.csv"
df = pd.read_csv(path)
df.head()

#Remove unnamed index column
df = df.drop("Unnamed: 0", axis = 1)

#Create X, y train/test sets
X = df.drop("Playoff", axis = 1)
y = df["Playoff"]

y = pd.DataFrame(y)
y["Year"] = X["Year"]

X.head()

#Train = before 2022
#Test = 2022
xTrain = X.loc[X["Year"] < 2022]
yTrain = y.loc[y["Year"] < 2022]

xTest = X.loc[X["Year"] == 2022]
yTest = y.loc[y["Year"] == 2022]

xTrain.columns
def dropYear(df):
    return df.drop("Year", axis = 1)

xTrain = dropYear(xTrain)
yTrain = dropYear(yTrain)
xTest = dropYear(xTest)
yTest = dropYear(yTest)

#Import rf classifier
from sklearn.ensemble import RandomForestClassifier

#Initialize and fit classifier
randForest = RandomForestClassifier(random_state = 0, max_depth = 2)
randForest.fit(xTrain, yTrain)

#Evaluate basic random forest
predictions = randForest.predict(xTest)

#Base rf does WORSE than pruned decision tree -- use CV to optimize parameters?
print(f"Random forest classifier test accuracy: {randForest.score(xTest, yTest)}")

print(f"Random forest classifier training accuracy: {randForest.score(xTrain, yTrain)}.")

#Look at hyperparameters
randForest.get_params()

#Try ccp alpha value that worked for decision trees?
randForest = RandomForestClassifier(random_state = 0, ccp_alpha=0.015)
randForest.fit(xTrain, yTrain)

#Slightly better already! -- 0.78125
print(f"Random forest classifier with ccp = 0.015 test accuracy: {randForest.score(xTest, yTest)}.")

#Create CV randomized grid search to tune hyperparameters
n_estimators = [int(x) for x in np.linspace(200, 2000, num = 10)]
max_features = ["sqrt", "log2", "None"]
bootstrap = [True, False]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

randomGrid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

from sklearn.model_selection import RandomizedSearchCV

randomForest = RandomForestClassifier()

randomGridForest = RandomizedSearchCV(estimator = randomForest, param_distributions = randomGrid, n_iter = 50, cv = 3, verbose = 2, random_state = 0, n_jobs = -1)

randomGridForest.fit(xTrain, yTrain)

randomGridForest.best_params_

#Let's see our new testing accuracy after the random search
randomForest = RandomForestClassifier(random_state=0, n_estimators=1200,
                                      min_samples_split=5,
                                      min_samples_leaf=2,
                                      max_features="sqrt",
                                      max_depth=110,
                                      bootstrap=True)

randomForest.fit(xTrain, yTrain)

#0.78125 - same as with optimal ccp -- still worse than DT?
print(f"Randomized search rf test accuracy: {randomForest.score(xTest, yTest)}.")

#Looks like it is overfitting by a lot though, 0.99 training accuracy 
print(f"Randomized search rf train accuracy: {randomForest.score(xTrain, yTrain)}.")

'''
I could continue with a grid search to further optimize, 
but it appears that the data is stable enough that a simple DT can work 
if pruned. A DT also allows for quick predicition and inference so
I will stick with that for now. 
'''

