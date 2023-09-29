import pandas as pd
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult=pd.read_csv('adult.data', names=headers, skipinitialspace=True)
adultTest=pd.read_csv('updateAdult.test', names=headers, skipinitialspace=True)
print("Total numer of test case: " + str(len(adultTest)))

def dataCleaningAndPreprocessing(dataset):
    newDataSet = dataset
    newDataSet=newDataSet.drop(['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'],axis=1)
    return newDataSet


updateAdult = dataCleaningAndPreprocessing(adult)
updateAdultTest = dataCleaningAndPreprocessing(adultTest)

##########################
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text
import numpy as np
import time

x_train = updateAdult.drop('income', axis=1)
y_train = updateAdult['income']
x_test = updateAdultTest.drop('income', axis=1)
y_test = updateAdultTest['income']

encoder = OrdinalEncoder()
startTime = time.time()
x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train_encoded, y_train)

train_predictions_tree = dtc.predict(x_train_encoded)
test_predictions_tree = dtc.predict(x_test_encoded)
train_accuracy_tree = accuracy_score(y_train, train_predictions_tree)
test_accuracy_tree = accuracy_score(y_test, test_predictions_tree)

endTime = time.time()
timeSpent = endTime - startTime
print(f"Time spent: {timeSpent} seconds")
print("Decision Tree - Train accuracy: ", train_accuracy_tree)
print("Decision Tree - Test accuracy: ", test_accuracy_tree)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, test_predictions_tree)
print(cm)
print(classification_report(y_test, test_predictions_tree, digits=4))
