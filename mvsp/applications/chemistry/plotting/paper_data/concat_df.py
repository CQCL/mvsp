import pandas as pd
import glob

# Get a list of all CSV files in the current directory
csv_files = glob.glob("*.csv")

# List to hold the DataFrames
dfs = []

# Loop through the CSV files and append each one to the list of DataFrames
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Concatenate the DataFrames
df_concat = pd.concat(dfs, ignore_index=True)

# Write the concatenated DataFrame to a new CSV file
df_concat.to_csv("concatenated.csv", index=False)
