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

xTrain.columns
def dropYear(df):
    return df.drop("Year", axis = 1)

xTrain = dropYear(xTrain)
yTrain = dropYear(yTrain)
xTest = dropYear(xTest)
yTest = dropYear(yTest)

#Import logistic regression modules
from sklearn.utils import shuffle
import statsmodels.api as sm

#Perform recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(xTrain, yTrain.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

rfeFilter = rfe.support_
xTrainFiltered = xTrain.iloc[:,rfeFilter]

#Generate and fit logit model
logitModel=sm.Logit(yTrain, xTrainFiltered)
logitResult=logitModel.fit()

print(logitResult.summary2())

logitResult.pvalues
sigVars = ['off_pts_g', 'punt_avg', 'record', 'sos']

#Create sklearn logistic regression to evaluate
logregModel = LogisticRegression()
logregModel.fit(xTrainFiltered, yTrain)

#Since we are mainly concerned with prediction, let's use all vars
xTestFiltered = xTest.iloc[:,rfeFilter]
yPred = logregModel.predict(xTestFiltered)

#62% accuracy, slightly better than random guessing-- good baseline
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregModel.score(xTestFiltered, yTest)))

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yTest, yPred)
print(confusion_matrix)

#More eval metrics
from sklearn.metrics import classification_report
print(classification_report(yTest, yPred))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(yTest, logregModel.predict(xTestFiltered))
fpr, tpr, thresholds = roc_curve(yTest, logregModel.predict_proba(xTestFiltered)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




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
