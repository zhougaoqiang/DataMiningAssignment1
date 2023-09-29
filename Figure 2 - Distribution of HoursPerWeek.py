import pandas as pd
import matplotlib.pyplot as plt

# Load the data
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataFilename = 'adult.data'
data = pd.read_csv(dataFilename, names=headers, skipinitialspace=True)

# Set up the figure and axis
plt.figure(figsize=(20, 4))

# Define color properties
boxprops = dict(linestyle='-', linewidth=2, color='#ff9999')
medianprops = dict(linestyle='-', linewidth=2, color='red')
whiskerprops = dict(linestyle='-', linewidth=2, color='#66b2ff')
capprops = dict(linestyle='-', linewidth=2, color='Lightblue')
flierprops = dict(marker='*', markerfacecolor='#66b2ff', markersize=4)

# Create the boxplot with custom color settings
plt.boxplot(data['hours-per-week'], vert=False,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)

# Label and title the plot
plt.ylabel('Hours-per-Week')
plt.title('Distribution of Hours-per-Week')

# Set x-axis ticks
plt.xticks(range(0, data['hours-per-week'].max() + 1, 2))

# Save the figure
plt.savefig("Figure 2 - Distribution of Hours-per-Week")

# Show the plot
plt.show()
