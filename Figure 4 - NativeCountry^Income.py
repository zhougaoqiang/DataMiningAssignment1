import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
            below50k[i], below50k[i + 1] = below50k [i + 1], below50k[i]
            above50k[i], above50k[i + 1] = above50k [i + 1], above50k[i]
            count += 1
    if count == 0 :
        break;



ind = np.arange(labelsCount)
width = 0.4
fig, ax = plt.subplots(figsize=(20, 7))
bars1 = ax.bar(ind, below50k, width, label='<=50K', color='#ff9999')
bars2 = ax.bar(ind, above50k, width, label='>50K', bottom=below50k, color='#66b2ff')
ax.set_xlabel('Country')
ax.set_ylabel('Number & Income')
ax.set_title('Number of Individuals by Country and Income')
ax.set_xticks(ind)
ax.set_xticklabels(nativeCountries, rotation=60, ha="right")
ax.legend()

# Function to display the percentage on top of the bars for '<=50K'
def autolabel_below(rects):
    for rect in rects:
        height = rect.get_height()
        # percentage = 100 * height / total[rects.index(rect)]
        ax.annotate(height,
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Function to display the percentage on top of the bars for '>50K'
def autolabel_above(rects):
    for rect in rects:
        height = rect.get_height()
        # percentage = 100 * height / total[rects.index(rect)]
        ax.annotate(height,
                    xy=(rect.get_x() + rect.get_width() / 2, height + below50k[rects.index(rect)] - height/2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel_below(bars1)
autolabel_above(bars2)
# fig.tight_layout()
# plt.savefig("Figure 3 - number of native countries")
# # plt.show()


# plt.figure(figsize=(20, 7))
# plt.plot(nativeCountries, totalPercentage, "r", label="above50k")
# # plt.plot(ageLabels, percentageBeloe50K, "b", label="below50k")
# plt.xticks(rotation=90)
# plt.xlabel("Countries")
# plt.ylabel("High income percentage")
# plt.title("Income percentage/Countries")
# plt.tight_layout()
# plt.savefig("Figure 4 - Country VS High Income Percentage")
# # plt.show()

ax1 = ax.twinx()
ax1.plot(nativeCountries, totalPercentage, "r", label="above 50k percentage")
ax1.set_ylabel('High income percentage')
ax1.legend(loc='upper left')

# Adjust layout and save the combined figure
fig.tight_layout()
plt.savefig("Figure 4 - Native Country VS Income")
plt.show()