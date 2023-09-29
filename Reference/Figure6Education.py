import pandas as pd

headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult=pd.read_csv('adult.data', names=headers, skipinitialspace=True)

import numpy as np
import matplotlib.pyplot as plt

educationLabels = list(set(adult['education']))
labelsCount = len(educationLabels)
below50k = []
above50k = []
total = []

for educationLabel in educationLabels:
    income = adult.loc[adult['education'] == educationLabel]['income']
    b50 = income[income.str.strip() == '<=50K'].count()
    a50 = income[income.str.strip() == '>50K'].count()
    below50k.append(b50)
    above50k.append(a50)
    total.append(b50 + a50)

ind = np.arange(labelsCount)
width = 0.6

fig, ax = plt.subplots(figsize=(20, 10))

bars1 = ax.bar(ind, below50k, width, label='<=50K', color='#ff9999')
bars2 = ax.bar(ind, above50k, width, label='>50K', bottom=below50k, color='#66b2ff')

ax.set_xlabel('Education')
ax.set_ylabel('Percentage of Individuals (%)')
ax.set_title('Percentage of Individuals by Education Year and Income')
ax.set_xticks(ind)
ax.set_xticklabels(educationLabels, rotation=45, ha="right")
ax.legend()

# Function to display the percentage on top of the bars for '<=50K'
def autolabel_below(rects):
    for rect in rects:
        height = rect.get_height()
        percentage = 100 * height / total[rects.index(rect)]
        ax.annotate('{:.1f}%'.format(percentage),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Function to display the percentage on top of the bars for '>50K'
def autolabel_above(rects):
    for rect in rects:
        height = rect.get_height()
        percentage = 100 * height / total[rects.index(rect)]
        ax.annotate('{:.1f}%'.format(percentage),
                    xy=(rect.get_x() + rect.get_width() / 2, height + below50k[rects.index(rect)] - height/2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel_below(bars1)
autolabel_above(bars2)

fig.tight_layout()
plt.savefig("Figure 6 - Education vs Income")
plt.show()
