import pandas as pd

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
dataFilename = 'adult.data'
df = pd.read_csv(dataFilename, names=headers, skipinitialspace=True)
print(df.shape)

df = df.loc[ (df['workclass'] != '?') & (df['occupation'] != '?') & (df['native-country']!= '?')]
print(df.shape)