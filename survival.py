#Finding out Accuracy of Titanic Survivors by Ensembling Methods
##Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import pickle
from yellowbrick.classifier import ConfusionMatrix, ROCAUC

##Reading CSV Files and it's information
dF = pd.read_csv("datasets/titanic.csv")
print(dF.head())
print(dF.describe())

##Finding out NULL values
print(dF.isnull().sum())

print(dF.isnull().sum(axis = 1).loc[:10])

mask = dF.isnull().any(axis = 1)
print(mask.head())

print(dF.Sex.value_counts(dropna = False))

print(dF.Embarked.value_counts(dropna = False))

##Removing Unecessary Columns
dF = dF.drop(columns = ["Name", "Ticket", "Fare", "Cabin", "Embarked"])
dF = pd.get_dummies(dF)
print(dF.columns)

##Cleaning Data
dF = dF.dropna(axis = 'rows')

##Passing Data for training and testing
x = dF.drop(columns = 'Survived')
y = dF['Survived']
 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=0)

num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
x_test_df = x_test.copy()

##Normalizing Data
cols = "'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch','Sex_female', 'Sex_male'".split (",")
sca = StandardScaler()
x_train = sca.fit_transform(x_train)
x_train = pd.DataFrame (x_train, columns = cols)
x_test = sca.transform(x_test)
x_test = pd.DataFrame (x_test, columns = cols)

#Using Logistic Regression, Decision Tree and SVC for Building different Models
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

##Using K Fold splitting
seed = 7
kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
num_trees = 100
max_features = x_train.shape[1]
estimators = []

model1 = LogisticRegression()
estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))

model3 = SVC()
estimators.append(('svm', model3))

classifier = VotingClassifier(estimators)
classifier.fit(x_train, y_train)

##Saving Model
filename = 'savedmodel.mod'
fp=open(filename,"wb")

##Using Pickle
pickle.dump(classifier,fp)
fp.close()
fp=open(filename,"rb")
classifier2 = pickle.load(fp)
y_pred= classifier2.predict(x_test)

##Making Confusion Matrix
cm= confusion_matrix(y_true=y_test, y_pred=y_pred)
##print(cm)

##Plotting Confusion Matrix
##mapping = {0: "died", 1: "survived"}
##fig, ax = plt.subplots (figsize = (6, 6))
##cm_viz = ConfusionMatrix (classifier2, classes = ["died", "survived"], label_encoder = mapping)
##cm_viz.score (x_test, y_test)
##cm_viz.poof ()

##Measuring Accuracy of Model
accu=accuracy_score(y_test, y_pred)
print("Accuracy is ",accu*100)

##Predictions
result = pd.DataFrame({"PassengerId": x_test_df['PassengerId'], "Survived": y_pred})
result = result.sort_values(by ='PassengerId', ascending = 1)
result.to_csv('result.csv', index=False)
