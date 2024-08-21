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

#Import multilayer preceptron classifier
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(xTrain, yTrain)
predictions = clf.predict(xTest)

clf.score(xTest, yTest) #56.25%

#Try grid search tuning
mlp_gs = MLPClassifier(max_iter=2)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(xTrain, yTrain) # X is train samples and y is the corresponding labels
clf.best_params_

#Rerun with best params
clf = MLPClassifier(activation='tanh', alpha=0.05, 
                    hidden_layer_sizes=(20,),
                    learning_rate='adaptive',
                    solver='adam', random_state=1)
clf.fit(xTrain, yTrain)

clf.score(xTest, yTest) #68.75% -- better but DT still outperforms