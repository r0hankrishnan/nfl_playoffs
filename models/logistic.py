#Article
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8



# Import basic modules
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

#Create training and testing sets
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


def dropYear(df):
    return df.drop("Year", axis = 1)

xTrain = dropYear(xTrain)
yTrain = dropYear(yTrain)
xTest = dropYear(xTest)
yTest = dropYear(yTest)

#Import logistic regression modules
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Generate and fit probit model
logit = sm.Logit(yTrain, xTrain)
results = logit.fit()

'''
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
'''
