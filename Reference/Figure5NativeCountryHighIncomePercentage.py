import pandas as pd
import matplotlib.pyplot as plt

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult=pd.read_csv('adult.data', names=headers, skipinitialspace=True)

nativeCountries = list(set(adult['native-country']))
print(nativeCountries)
labelsCount = len(nativeCountries)
below50k = []
above50k = []
total =[]
totalPercentage  = []

for country in nativeCountries:
    income = adult.loc[adult['native-country'] == country]['income']
    b50 = income[income.str.strip() == '<=50K'].count()
    a50 = income[income.str.strip() == '>50K'].count()
    below50k.append(b50)
    above50k.append(a50)
    total.append(a50 + b50)
    totalPercentage.append(a50/(a50+b50))
    
len = len(totalPercentage)

#bubble sort
for j in range (len-1):
    count = 0
    for i in range (0, len-1-j) :
        if totalPercentage[i] > totalPercentage[i + 1] :
            totalPercentage[i], totalPercentage[i + 1] = totalPercentage [i + 1], totalPercentage[i]
            nativeCountries[i], nativeCountries[i + 1] = nativeCountries [i + 1], nativeCountries[i]
            count += 1
    if count == 0 :
        break;

plt.figure(figsize=(20, 7))
plt.plot(nativeCountries, totalPercentage, "r", label="above50k")
# plt.plot(ageLabels, percentageBeloe50K, "b", label="below50k")
plt.xticks(rotation=90)
plt.xlabel("Countries")
plt.ylabel("High income percentage")
plt.title("Income percentage/Countries")
plt.tight_layout()
plt.savefig("Figure 5 - Country VS High Income Percentage")
plt.show()