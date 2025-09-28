# In update_tickers.py (Final Working Version)

import pandas as pd
import requests
from io import StringIO

def get_nifty50_tickers():
    """
    Scrapes the Wikipedia page for the list of NIFTY 50 companies
    and returns a list of yfinance-compatible tickers.
    """
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        
        print("Fetching the latest list of NIFTY 50 companies from Wikipedia...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        all_tables = pd.read_html(StringIO(response.text))
        
        # --- THE FIX ---
        # Based on our debugging, the correct table is at index 1
        nifty50_df = all_tables[1]
        
        if 'Symbol' in nifty50_df.columns:
            tickers = nifty50_df['Symbol'].tolist()
            nifty_tickers_ns = [f"{ticker}.NS" for ticker in tickers]
            print("Successfully fetched and formatted tickers.")
            return nifty_tickers_ns
        else:
            print("Error: Could not find the 'Symbol' column in the table.")
            return None
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    latest_tickers = get_nifty50_tickers()
    if latest_tickers:
        print("\n--- Latest NIFTY 50 Tickers ---")
        print(latest_tickers)
        print(f"\nTotal tickers found: {len(latest_tickers)}")