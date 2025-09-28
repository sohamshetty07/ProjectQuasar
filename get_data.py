import yfinance as yf
import pandas as pd
import os
from update_tickers import get_nifty50_tickers # Import our function

# --- Configuration ---
DATA_DIR = "data"
PERIOD = "5y"

# --- Main Script ---

# 1. Get the latest list of tickers automatically
NIFTY_50_TICKERS = get_nifty50_tickers()

# 2. Check if the ticker list was successfully fetched
if NIFTY_50_TICKERS:
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # 3. Loop through each ticker and download its data
    for ticker_symbol in NIFTY_50_TICKERS:
        try:
            print(f"Downloading data for {ticker_symbol}...")
            
            # Create a ticker object
            ticker_data = yf.Ticker(ticker_symbol)
            
            # Get historical data for the defined period
            historical_data = ticker_data.history(period=PERIOD)
            
            # Define the full path to save the file
            file_name = f"{ticker_symbol}_historical_data.csv"
            file_path = os.path.join(DATA_DIR, file_name)
            
            # Save the data to a CSV file in the 'data' directory
            historical_data.to_csv(file_path)
            
            print(f"--> Success! Data for {ticker_symbol} saved to {file_path}")
            
        except Exception as e:
            print(f"--> FAILED to download data for {ticker_symbol}. Error: {e}")

    print("\n--- All data downloads complete. ---")
else:
    print("Could not fetch the ticker list. Halting data download.")