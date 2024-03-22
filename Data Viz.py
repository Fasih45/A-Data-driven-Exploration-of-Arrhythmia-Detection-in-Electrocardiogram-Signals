import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Medical Diagonsis.csv')

# Preprocess the data if necessary
# For example, convert age column to numeric if it's stored as string

# Plot box plots or violin plots for each diagnosis
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, x='DX', y='Age')
plt.xticks(rotation=45)
plt.title('Age Distribution for Each Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Age')
plt.tight_layout()
plt.show()

# Optionally, you can also use violin plots for a more detailed distribution
plt.figure(figsize=(12, 8))
sns.violinplot(data=data, x='DX', y='Age')
plt.xticks(rotation=45)
plt.title('Age Distribution for Each Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Age')
plt.tight_layout()
plt.show()
