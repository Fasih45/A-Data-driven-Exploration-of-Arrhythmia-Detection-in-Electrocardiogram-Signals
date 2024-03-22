import pandas as pd

# Define the mapping dictionary from full category names to acronyms
dx_mapping_acronyms = {
    'atrial fibrillation': 'AF',
    'general supraventricular tachycardia': 'GST',
    'sinus bradycardia': 'SB',
    'sinus rhythm including sinus irregularity rhythm': 'SR',
    'sinus tachycardia': 'ST'
}

# Read data from CSV into a DataFrame
data = pd.read_csv('output.csv')

# Replace the full category names with acronyms
data.replace({"DX": dx_mapping_acronyms}, inplace=True)

# Write the modified data back to a file
data.to_csv('Medical Diagonsis.csv', index=False)
