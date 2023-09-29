import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataFilename = 'adult.data'
data = pd.read_csv(dataFilename, names=headers, skipinitialspace=True)
age = data['age']
ageLabels = sorted(list(set(age)))

# Calculate the counts and percentages for each age
below50k = []
above50k = []
total = []
percentageAbove50K = []
for ageLabel in ageLabels:
    income = data.loc[data['age'] == ageLabel]['income']
    b50 = income[income.str.strip() == '<=50K'].count()
    a50 = income[income.str.strip() == '>50K'].count()
    below50k.append(b50)
    above50k.append(a50)
    total.append(b50 + a50)
    percentageAbove50K.append(a50 / (b50 + a50))

# Create the bar plot
ageInd = np.arange(len(total))
width = 0.3
fig, ax1 = plt.subplots(figsize=(20, 7))
bars1 = ax1.bar(ageInd, below50k, width, label='<=50K', color='#ff9999')
bars2 = ax1.bar(ageInd, above50k, width, label='>50K', bottom=below50k, color='#66b2ff')

# Set labels and title for the bar plot
ax1.set_xlabel('Age')
ax1.set_ylabel('Number of >50K and <50K')
ax1.set_title('Age VS Income and Income Percentage')
ax1.set_xticks(ageInd)
ax1.set_xticklabels(ageLabels, rotation=45, ha="right")
ax1.legend(loc="upper left")

# Create the line plot
ax2 = ax1.twinx()
ax2.plot(ageInd, percentageAbove50K, "r", label="Percentage Above 50K")
ax2.set_ylabel('Income Percentage')
ax2.legend(loc="upper right")

# Save and show the plot
fig.tight_layout()
plt.savefig("Figure 1 - Age VS Income and Income")
plt.show()
