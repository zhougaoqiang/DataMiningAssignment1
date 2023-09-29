import pandas as pd
headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

dataFilename = 'adult.data'
data = pd.read_csv(dataFilename, names=headers, skipinitialspace=True)

capitalGain = data['capital-gain']
capitalLoss = data['capital-loss']
capitalProfit = []
income = []
dataLen = len(data['income'])
i = 0
while i < dataLen:
   if (data['income'][i] == '>50K'):
      income.append(1)
   else:
      income.append(0)
   i = i + 1
data['income'] = income
correlation = data['capital-gain'].corr(data['income'] )
print("capital-gain vs income: correaltion:" + str(correlation))

correlation = data['capital-loss'].corr(data['income'] )
print("capital-loss vs income: correaltion:" + str(correlation))


i = 0
while i < dataLen:
   capitalProfit.append(capitalGain[i] - capitalLoss[i])
   i = i + 1

newdata = {'capital' : capitalProfit, 'income' : income}
df = pd.DataFrame(newdata);
correlation = df['capital'].corr(df['income'])
print("capital vs income: correaltion: " + str(correlation))


print(len(data.columns))