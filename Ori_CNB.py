import pandas as pd
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult=pd.read_csv('adult.data', names=headers, skipinitialspace=True)
adultTest=pd.read_csv('updateAdult.test', names=headers, skipinitialspace=True)

print("Total numer of test case: " + str(len(adultTest)))

def dataCleaningAndPreprocessing(dataset):
    newDataSet = dataset
    newDataSet=newDataSet.drop(['age','fnlwgt','education-num', 'capital-gain','capital-loss','hours-per-week'],axis=1)
    return newDataSet


updateAdult = dataCleaningAndPreprocessing(adult)
updateAdultTest = dataCleaningAndPreprocessing(adultTest)

##########################
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import time
model = CategoricalNB(alpha=1)
encoder = OrdinalEncoder()

x_train = updateAdult.drop('income', axis=1)
y_train = updateAdult['income']
x_test = updateAdultTest.drop('income', axis=1)
y_test = updateAdultTest['income']

# Fit and transform on the training data
startTime = time.time()
x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

# Train the model using the training sets
gnb = model.fit(x_train_encoded,y_train)
train_predictions = gnb.predict(x_train_encoded)
train_accuracy = accuracy_score(y_train, train_predictions)
test_predictions = gnb.predict(x_test_encoded)
test_accuracy = accuracy_score(y_test,test_predictions)

#print result
endTime = time.time()
timeSpent = endTime - startTime
print(f"Time spent: {timeSpent} seconds")
print("train accuracy: ", train_accuracy)
print("test accurary: ", test_accuracy)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, test_predictions)
print(cm)
print(classification_report(y_test, test_predictions, digits=4))