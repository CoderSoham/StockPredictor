import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv('goog.csv')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Ensure consistent date format
data['Date'] = pd.to_datetime(data['Date'])

# Handle outliers (if necessary)
# For example, remove rows with closing prices greater than a certain threshold
data = data[data['Close'] < 10000]  # Adjust the threshold as needed

# Rename columns (if necessary)
# For example, rename 'Close' column to 'Closing_Price'
data.rename(columns={'Close': 'Closing_Price'}, inplace=True)

# Check the cleaned DataFrame
print("\nCleaned Data Info:")
print(data.info())

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_goog.csv', index=False)
