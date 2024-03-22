import numpy as np
import csv
import math
import sys
import os

def NLM_1dDarbon(signal, Nvar, P, PatchHW):
    """
    Perform 1D non-local means (NLM) denoising on a signal.

    Parameters:
    - signal: The input signal to be denoised.
    - Nvar: The noise standard deviation of the signal.
    - P: The patch size for denoising.
    - PatchHW: Half of the patch size.

    Returns:
    - denoisedSig: The denoised signal.
    """
    if isinstance(P, int):
        P = P - 1
        Pvec = np.array(range(-P, P + 1))
    else:
        Pvec = P

    signal = np.array(signal)
    N = len(signal)

    denoisedSig = np.empty(len(signal))
    denoisedSig[:] = np.nan

    iStart = PatchHW + 1
    iEnd = N - PatchHW
    denoisedSig[iStart: iEnd] = 0

    Z = np.zeros(len(signal))

    Npatch = 2 * PatchHW + 1
    h = 2 * Npatch * Nvar ** 2

    for idx in Pvec:
        k = np.array(range(N))
        kplus = k + idx
        igood = np.where((kplus >= 0) & (kplus < N))
        SSD = np.zeros(len(k))
        SSD[igood] = (signal[k[igood]] - signal[kplus[igood]]) ** 2
        Sdx = np.cumsum(SSD)

        for ii in range(iStart, iEnd):
            distance = Sdx[ii + PatchHW] - Sdx[ii - PatchHW - 1]
            w = math.exp(-distance / h)
            t = ii + idx

            if t > 0 and t < N:
                denoisedSig[ii] = denoisedSig[ii] + w * signal[t]
                Z[ii] = Z[ii] + w

    denoisedSig = denoisedSig / (Z + sys.float_info.epsilon)
    denoisedSig[0: PatchHW + 1] = signal[0: PatchHW + 1]
    denoisedSig[-PatchHW:] = signal[-PatchHW:]

    return denoisedSig


def read_csv(file_path):
    """
    Read data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - headers: List of column headers.
    - data: 2D array containing the data from the CSV file.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header row
        for row in reader:
            data.append([float(value) for value in row])
    return headers, np.array(data)


def write_csv(file_path, data, headers=None):
    """
    Write data to a CSV file.

    Parameters:
    - file_path: Path to the CSV file to be written.
    - data: 2D array containing the data to be written.
    - headers: Optional list of column headers.
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)

# Directory containing the CSV files to be denoised
input_directory = r'D:\University\8th semester\Data Science\Assignment no 3\results\WFDBRecords'

# Directory to store denoised results
output_directory = 'denoised_results'
os.makedirs(output_directory, exist_ok=True)

# Traverse through all directories and subdirectories
for root, dirs, files in os.walk(input_directory):
    for file_name in files:
        if file_name.endswith('.csv'):
            # Construct full path to the input CSV file
            input_file_path = os.path.join(root, file_name)

            # Load the data and headers from the CSV file
            headers, data = read_csv(input_file_path)

            # Initialize a list to store denoised data for each lead
            denoised_data = []

            # Apply NLM Denoising to Each Lead
            num_columns = len(data[0])
            for col_idx in range(1, num_columns):
                signal = data[:, col_idx]
                Nvar = np.std(signal)
                P = 3
                PatchHW = 1
                denoised_signal = NLM_1dDarbon(signal, Nvar, P, PatchHW)
                denoised_data.append(denoised_signal)

            # Combine Denoised Data
            denoised_data = np.array(denoised_data).T
            denoised_data_with_time = np.column_stack((data[:, 0], denoised_data))

            # Write Denoised Data to CSV
            output_file_path = os.path.join(output_directory, f'denoised_{file_name}')
            write_csv(output_file_path, denoised_data_with_time, headers)

            print(f"Denoising complete. Denoised signal saved to {output_file_path}")
