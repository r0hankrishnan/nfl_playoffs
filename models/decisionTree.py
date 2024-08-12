#Documentation
#https://scikit-learn.org/stable/modules/tree.html#classification


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

#Import decision tree classifier
from sklearn import tree

#Initialize
treeModel = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 5)
treeModel = treeModel.fit(xTrain, yTrain)

#Visualize Tree
tree.plot_tree(treeModel)

#Predict
predictions = treeModel.predict(xTest)

#Evaluate -- better than logistic regression
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(treeModel.score(xTest, yTest)))

#Can we improve this model with cost complexity tuning?
#Implement ccp tuning using sci-kit learn documentation
tuningTree = tree.DecisionTreeClassifier(random_state=0)
path = tuningTree.cost_complexity_pruning_path(xTrain, yTrain)
ccpAlphas, impurities = path.ccp_alphas, path.impurities

#Plot impurity as alpha increases
fig, ax = plt.subplots()
ax.plot(ccpAlphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("Effective Alpha")
ax.set_ylabel("Total Impurity of Leaves")
ax.set_title("Total Impurity vs Effective Alpha for Training Set")

#Test all alphas to see which performs best on test
models = []
for ccp_alpha in ccpAlphas:
    model = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    model.fit(xTrain, yTrain)
    models.append(model)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        models[-1].tree_.node_count, ccpAlphas[-1]
    )
)

#Plot depth and nodes against alpha
models = models[:-1] #Choosing all models except final -- only one obs per leaf
ccp_alphas = ccpAlphas[:-1] #Same as above

nodeCounts = [model.tree_.node_count for model in models]
depth = [model.tree_.max_depth for model in models]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, nodeCounts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Number of Nodes")
ax[0].set_title("Number of Nodes vs Alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

#Plot models to compare training and test errors - 0.015 performs best
trainScores = [model.score(xTrain, yTrain) for model in models]
testScores = [model.score(xTest, yTest) for model in models]

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
ax.plot(ccp_alphas, trainScores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, testScores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

#Final model
treeFinal = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.015)
treeFinal = treeFinal.fit(xTrain, yTrain)

#Accuracy
print(f"Accuracy of DTC on test: {treeFinal.score(xTest, yTest)}.")

#ROC curve -- lots of room for improvement
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
treeROCAUC = roc_auc_score(yTest, treeFinal.predict(xTest))
fpr, tpr, thresholds = roc_curve(yTest, treeFinal.predict_proba(xTest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree Classifier (area = %0.2f)' % treeROCAUC)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("../assets/treeROC")
plt.show()

