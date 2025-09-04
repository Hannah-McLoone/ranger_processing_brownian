import pandas as pd
import numpy as np
import glob

# Step 1: Get all CSV files in the folder (adjust path as needed)
csv_files = glob.glob('/maps/hm708/bb_maps/oban*_intensity90.csv')  # e.g., "./csv_files/*.csv"


# Step 2: Initialize sum array as None
sum_grid = None

# Step 3: Loop through files and sum
for file in csv_files:
    df = pd.read_csv(file, header=None)  # assuming no headers
    if sum_grid is None:
        sum_grid = df.values
    else:
        sum_grid += df.values

# Step 4: Convert the sum back to DataFrame
sum_df = pd.DataFrame(sum_grid)

# Step 5: Save the summed grid to a new CSV
sum_df.to_csv("sanity_check.csv", index=False, header=False)

print("Summed CSV saved as 'summed_grid.csv'.")
