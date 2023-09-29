import pandas as pd

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult=pd.read_csv('adult.data', names=headers, skipinitialspace=True)
adultTest=pd.read_csv('updateAdult.test', names=headers, skipinitialspace=True)

def handleUnknownNativeCountry(country) :
    if country == '?' :
        return 'United-States'
    else : 
        return country

#data-preprocessing change continues to discrete
#process age
def convertAge(age):
    if age < 22 :
        return '<22'
    elif age >=72 :
        return '>=72'
    else :
        base_age = (age - 22) // 5
        start_age = 22 + base_age * 5
        end_age = 22 + 4 + base_age * 5
        return f"{start_age}-{end_age}"

#process capital-gain and capital-loss
def convertCapital(capitalGain, capitalLoss):
    dataLen = len(capitalGain)
    if dataLen != len(capitalLoss):
        raise AssertionError("Lengths of capitalGain and capitalLoss are not equal.")
    i = 0
    capitalOptimized = []
    while i < dataLen:
        profit = capitalGain[i] - capitalLoss[i]
        if(profit == 0):
            capitalOptimized.append('No captial profit')
        elif (profit < -3000):
             capitalOptimized.append('<-3000')
        elif (profit < 0):
            capitalOptimized.append('>-3000&<0')
        elif (profit <3000):
            capitalOptimized.append('>0&<3000')
        elif (profit < 6000):
            capitalOptimized.append('>=3000&<6000')
        else:
            capitalOptimized.append('>=6000')
        i = i + 1
    return capitalOptimized

# def combineHoursPerWeek(hoursPerWeek) :
#     combineHoursPerWeekData = []
#     for hoursPerWeekData in hoursPerWeek :
#         hours = '=40'
#         if hoursPerWeekData < 40 :
#             hours = '<40'
#         elif hoursPerWeekData > 40 :
#             hours = '>40'
#         combineHoursPerWeekData.append(hours)
#     return combineHoursPerWeekData

def combineHoursPerWeek(hoursPerWeek) :
    combineHoursPerWeekData = []
    for hoursPerWeekData in hoursPerWeek :
        if hoursPerWeekData < 33:
            hours = "<33"
        elif hoursPerWeekData < 40 :
            hours = '>=33 & <40'
        elif hoursPerWeekData < 45 :
            hours = ">=40 & <45"
        elif hoursPerWeekData < 52 :
            hours = ">=45 & <52"
        else :
            hours = '>=52'
        combineHoursPerWeekData.append(hours)
    return combineHoursPerWeekData

countryGroups = {
    "Local" : ['United-States','?', 'South'],
    "North America": ['Canada', 'Outlying-US(Guam-USVI-etc)', 'Puerto-Rico', 'Mexico'],
    "Europe": ['Holand-Netherlands', 'England', 'Ireland', 'France', 'Yugoslavia', 'Scotland', 'Portugal', 
               'Germany', 'Greece', 'Italy', 'Poland', 'Hungary'],
    "Asia": ['Laos', 'India', 'Philippines', 'Hong', 'Japan', 'Cambodia', 'China', 'Taiwan', 'Iran', 'Thailand', 'Vietnam'],
    "Other America": ['Jamaica', 'Cuba', 'Dominican-Republic', 'Haiti', 'Trinadad&Tobago', 'Guatemala', 'Honduras', 'El-Salvador', 'Nicaragua','Ecuador', 'Columbia', 'Peru'],
}
def countryToArea(country):
    for group, countries in countryGroups.items():
        if country in countries:
            return group
    return "Unknown"  # no one will call this, but for in case

educationLevel = {
    "Primary" : ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'],
    "Middle" : ['HS-grad','Some-college'],
    "Associate" : ['Assoc-voc', 'Assoc-acdm'],
    "Senior": ['Bachelors', 'Masters'],
    "Professional": ['Doctorate', 'Prof-school']
} 
def convertToEducationLevel(education):
    for level, education in educationLevel.items() :
        if level in education :
            return level
    return "Unknown"  # no one will call this, but for in case

def dataCleaningAndPreprocessing(dataset, cleanMissingData):
    newDataSet = dataset
    newDataSet=newDataSet.drop(['fnlwgt','education-num','relationship'],axis=1) #axis = 1 means drop column, axis = 0 => drop row
    newDataSet['age'] = newDataSet['age'].apply(convertAge)
    newDataSet['capital-gain'] = convertCapital(newDataSet['capital-gain'], newDataSet['capital-loss'])
    newDataSet=newDataSet.drop('capital-loss',axis=1)
    newDataSet['hours-per-week'] = combineHoursPerWeek(newDataSet['hours-per-week'])
    if cleanMissingData == True: 
        newDataSet = newDataSet.loc[ (newDataSet['workclass'] != '?') & (newDataSet['occupation'] != '?') & (newDataSet['native-country']!= '?')]
    else :
        # newDataSet['native-country'] = newDataSet['native-country'].apply(handleUnknownNativeCountry)
        # tread '?' as a group in workclass and occupation, because may have some reasons such as confidential, indescribable
        newDataSet['area'] = newDataSet['native-country'].apply(countryToArea)
        newDataSet = newDataSet.drop('native-country', axis=1)
    # newDataSet['education'] = newDataSet['education'].apply(convertToEducationLevel)
    return newDataSet

def removeHeaders(headers):
    headers.remove('fnlwgt') 
    headers.remove('education-num')
    headers.remove('relationship')
    headers.remove('capital-loss')
    headers.remove('native-country')
    return headers
    
updateAdult= dataCleaningAndPreprocessing(adult, False)
updateAdultTest= dataCleaningAndPreprocessing(adultTest, False)
headers = removeHeaders(headers)
print(len(updateAdult.columns))

########data-balancing for original data  ==> tested useless
# from sklearn.utils import resample
# majority = updateAdult[updateAdult.income == '<=50K'];
# minority = updateAdult[updateAdult.income == '>50K'];
# print(len(majority));
# print(len(minority));

# minority = resample(minority, replace=True, n_samples=len(majority))
# updateAdult = pd.concat([majority, minority])

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

#get Best criterion, max_depth, min_samples_leaf, and min_samples_split
prepruningStartTime = time.time()
dtc = DecisionTreeClassifier(random_state=0)
param_grid = {
    # 'criterion': ['gini', "entropy"],
    'max_depth': [8,9,10,11,13,15],
    'min_samples_split': range(5,20,4), 
    'min_samples_leaf': range(3,10,1)
}
gridSearch = GridSearchCV(dtc, 
                           param_grid,
                           n_jobs=-1,
                           verbose=1,
                           cv=5)
gridSearch.fit(x_train_encoded, y_train)
best_params = gridSearch.best_params_
print(f'get best other params {best_params}')
prepruningEndTime = time.time()
timeSpent = prepruningEndTime - prepruningStartTime
print(f"Pre-Pruning Time spent: {timeSpent:.4f} seconds")

dtc = DecisionTreeClassifier(
                                max_depth=best_params.get('max_depth'), 
                                min_samples_leaf=best_params.get('min_samples_leaf'), 
                                min_samples_split=best_params.get('min_samples_split'),
                                # ccp_alpha=best_ccp_alpha
                            )
dtc.fit(x_train_encoded, y_train)
# r = export_text(dtc, feature_names=headers)
# print(r) #print tree
trainEndtime = time.time()
timeSpent = trainEndtime - startTime
print(f"Train Time spent: {timeSpent:.4f} seconds")

test_predictions_tree = dtc.predict(x_test_encoded)
preditEndtime = time.time()
timeSpent = preditEndtime - trainEndtime
print(f"Test Time spent: {timeSpent:.4f} seconds")
timeSpent = preditEndtime - startTime
print(f"Overall Time spent: {timeSpent:.4f} seconds")

train_predictions_tree = dtc.predict(x_train_encoded)
train_accuracy_tree = accuracy_score(y_train, train_predictions_tree)
test_accuracy_tree = accuracy_score(y_test, test_predictions_tree)
print(f"train accuracy:  {train_accuracy_tree:.4f}")
print(f"test accuracy:  {test_accuracy_tree:.4f}")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, test_predictions_tree)
#cm[0][0] = TP
#cm[1][1] = TN
#cm[0][1] = FP
#cm[1][0] = FN
print(cm)
print(classification_report(y_test, test_predictions_tree, digits=4))