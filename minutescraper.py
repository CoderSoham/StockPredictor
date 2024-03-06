from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime, timedelta

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = '1PA1CCOMB20XRMS3'

# Symbol of the stock (e.g., 'GOOGL' for Google)
symbol = 'GOOGL'

# Interval for the data (e.g., '1min' for 1-minute interval)
interval = '1min'

# Create a TimeSeries instance with the API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch intraday data until the current date
dataframes = []
start_date = datetime(2010, 1, 1)
end_date = datetime.today()
current_date = start_date

print("Fetching data...")
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    dataframes.append(data)

    # Update progress
    percent_complete = round((current_date - start_date).days / (end_date - start_date).days * 100, 2)
    print(f"\rProgress: {percent_complete}% complete", end="")

    current_date = next_date

# Concatenate all dataframes into a single dataframe
full_data = pd.concat(dataframes)

# Save the data to a CSV file
full_data.to_csv('goog_intraday_data_2010_to_current.csv')

print("\nData fetched and saved successfully.")
