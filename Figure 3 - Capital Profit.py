import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sb

# Load the data
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataFilename = 'adult.data'
data = pd.read_csv(dataFilename, names=headers, skipinitialspace=True)

capitalGain = data['capital-gain']
capitalLoss = data['capital-loss']
capitalProfit = []
dataLen = len(data['income'])
   
i = 0
# print(dataLen)
while i < dataLen:
    profit = capitalGain[i] - capitalLoss[i]
    if profit < 10000 and profit != 0: #remove >10000 and profit = 0 for better check other value's density
        capitalProfit.append(capitalGain[i] - capitalLoss[i])
    i = i + 1

# Set up the figure and axis
plt.figure(figsize=(20, 4))

# Define color properties
# boxprops = dict(linestyle='-', linewidth=2, color='#ff9999')
# medianprops = dict(linestyle='-', linewidth=2, color='red')
# whiskerprops = dict(linestyle='-', linewidth=2, color='#66b2ff')
# capprops = dict(linestyle='-', linewidth=2, color='Lightblue')
# flierprops = dict(marker='*', markerfacecolor='#66b2ff', markersize=2)
# plt.boxplot(capitalProfit, vert=False,
            #  boxprops=boxprops, medianprops=medianprops,
            # whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
#  
# plt.ylabel('Capital Profit')
# plt.xticks(range(-3000, 6000, 500))
# plt.title('Distribution of Capital Profit')
# plt.savefig("Figure 4 capital profit")
# plt.show()
# 
plt.scatter(capitalProfit, [0] * len(capitalProfit), alpha=0.1)
plt.xlabel('Capital Profit Distribution')
plt.title('Remove no profit and outliers(>10000) for better review')
plt.savefig("Figure 3 - Capital Profit")
plt.show()


#save_pic_filename='sns_kdeplot_2.png'
# sb.kdeplot(x=capitalProfit, fill=True, common_norm=False, alpha=.5, linewidth=0)
# plt.savefig("profit",dpi=600)
# plt.show()