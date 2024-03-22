import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Medical Diagonsis.csv')

# Preprocess the data
# Convert the DX column to a list of diagnosis codes
data['DX'] = data['DX'].apply(lambda x: x.split(','))

# Create a list of unique diagnosis codes
unique_diagnoses = sorted(set(diagnosis for sublist in data['DX'] for diagnosis in sublist))

# Initialize a co-occurrence matrix
co_occurrence_matrix = pd.DataFrame(0, index=unique_diagnoses, columns=unique_diagnoses)

# Update the co-occurrence matrix based on the data
for diagnoses in data['DX']:
    for diagnosis1 in diagnoses:
        for diagnosis2 in diagnoses:
            co_occurrence_matrix.loc[diagnosis1, diagnosis2] += 1

# Plot the co-occurrence matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_occurrence_matrix, cmap='coolwarm', annot=True, fmt='d')
plt.title('Co-occurrence of Diagnoses')
plt.xlabel('Diagnosis (Diagnosed Together)')
plt.ylabel('Diagnosis (Diagnosed Together)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

