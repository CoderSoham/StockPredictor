import pandas as pd

# Read the first few rows of data from CSV
data_head = pd.read_csv('4comp24yr.csv', nrows=5)

# Print the first few rows to inspect the DataFrame structure
print(data_head)
