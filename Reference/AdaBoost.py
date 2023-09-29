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

def combineHoursPerWeek(hoursPerWeek) :
    combineHoursPerWeekData = []
    for hoursPerWeekData in hoursPerWeek :
        hours = '=40'
        if hoursPerWeekData < 40 :
            hours = '<40'
        elif hoursPerWeekData > 40 :
            hours = '>40'
        combineHoursPerWeekData.append(hours)
    return combineHoursPerWeekData

countryGroups = {
    "Local" : ['United-States','?', 'South'],
    "North America": ['Canada', 'Outlying-US(Guam-USVI-etc)', 'Puerto-Rico', 'Mexico'],
    "Europe": ['Holand-Netherlands', 'England', 'Ireland', 'France', 'Yugoslavia', 'Scotland', 'Portugal', 
               'Germany', 'Greece', 'Italy', 'Poland', 'Hungary'],
    "Asia": ['Laos', 'India', 'Philippines', 'Hong', 'Japan', 'Cambodia', 'China', 'Taiwan', 'Iran', 'Thailand', 'Vietnam'],
    "Other America": ['Jamaica', 'Cuba', 'Dominican-Republic', 'Haiti', 'Trinadad&Tobago''Guatemala', 'Honduras', 'El-Salvador', 'Nicaragua','Ecuador', 'Columbia', 'Peru'],
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
    newDataSet['education'] = newDataSet['education'].apply(convertToEducationLevel)
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

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

x_train = updateAdult.drop('income', axis=1)
y_train = updateAdult['income']
x_test = updateAdultTest.drop('income', axis=1)
y_test = updateAdultTest['income']

encoder = OrdinalEncoder()
x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

from sklearn.ensemble import AdaBoostClassifier

# Keeping the previous data preprocessing steps as they are

# Use DecisionTreeClassifier as the base estimator for AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=2)  # Typically, AdaBoost uses "stumps" (trees with depth=1)

# Instantiate the AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=0)

# Fit the AdaBoost model
ada.fit(x_train_encoded, y_train)

# Predict on training and test data
train_predictions_ada = ada.predict(x_train_encoded)
test_predictions_ada = ada.predict(x_test_encoded)

# Compute accuracies
train_accuracy_ada = accuracy_score(y_train, train_predictions_ada)
test_accuracy_ada = accuracy_score(y_test, test_predictions_ada)

# Print results
print("AdaBoost - Train accuracy: ", train_accuracy_ada)
print("AdaBoost - Test accuracy: ", test_accuracy_ada)