# run_portfolio_forecast.py (Updated for Full NIFTY 50 Run)

from tqdm import tqdm
import traceback

from run_pipeline import run_feature_engineering_pipeline, run_model_training_pipeline
from update_tickers import get_nifty50_tickers

# A more comprehensive keyword map for finding relevant news.
# This is a starting point and can be refined over time.
STOCK_KEYWORDS_MAP = {
    "RELIANCE.NS": ["reliance", "ril", "jio", "ambani"], "TCS.NS": ["tcs", "tata consultancy"],
    "HDFCBANK.NS": ["hdfc bank"], "ICICIBANK.NS": ["icici bank"],
    "INFY.NS": ["infosys", "infy"], "HINDUNILVR.NS": ["hindustan unilever", "hul"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel"], "ITC.NS": ["itc limited"],
    "SBIN.NS": ["state bank of india", "sbi"], "LICI.NS": ["lic", "life insurance corporation"],
    "BAJFINANCE.NS": ["bajaj finance"], "HCLTECH.NS": ["hcl tech"], "LT.NS": ["larsen & toubro"],
    "KOTAKBANK.NS": ["kotak mahindra bank"], "ASIANPAINT.NS": ["asian paints"],
    "AXISBANK.NS": ["axis bank"], "MARUTI.NS": ["maruti suzuki"], "SUNPHARMA.NS": ["sun pharma"],
    "TITAN.NS": ["titan company"], "WIPRO.NS": ["wipro"], "TATAMOTORS.NS": ["tata motors", "tml", "jlr"],
    "ULTRACEMCO.NS": ["ultratech cement"], "ADANIENT.NS": ["adani enterprises"],
    "ONGC.NS": ["ongc", "oil and natural gas corporation"], "NTPC.NS": ["ntpc limited"],
    "JSWSTEEL.NS": ["jsw steel"], "POWERGRID.NS": ["power grid"], "COALINDIA.NS": ["coal india"],
    "M&M.NS": ["mahindra & mahindra", "m&m"], "TATASTEEL.NS": ["tata steel"],
    "HDFCLIFE.NS": ["hdfc life"], "BAJAJFINSV.NS": ["bajaj finserv"],
    "INDUSINDBK.NS": ["indusind bank"], "HINDALCO.NS": ["hindalco"],
    "TECHM.NS": ["tech mahindra"], "BRITANNIA.NS": ["britannia"], "NESTLEIND.NS": ["nestle india"],
    "GRASIM.NS": ["grasim"], "CIPLA.NS": ["cipla"], "SBILIFE.NS": ["sbi life insurance"],
    "EICHERMOT.NS": ["eicher motors", "royal enfield"], "DRREDDY.NS": ["dr reddy's"],
    "HEROMOTOCO.NS": ["heromotoco", "hero motocorp"], "APOLLOHOSP.NS": ["apollo hospitals"],
    "BAJAJ-AUTO.NS": ["bajaj auto"], "DIVISLAB.NS": ["divi's lab"],
    "SHRIRAMFIN.NS": ["shriram finance"], "LTIM.NS": ["ltimindtree"]
}

def main():
    print("--- 🚀 Starting Portfolio-Wide Model Training ---")
    
    tickers_to_process = get_nifty50_tickers()
    if not tickers_to_process:
        print("Could not fetch ticker list. Exiting.")
        return
        
    for ticker in tqdm(tickers_to_process, desc="Overall Progress"):
        try:
            keywords = STOCK_KEYWORDS_MAP.get(ticker, [ticker.split('.')[0].lower()])
            
            analysed_data = run_feature_engineering_pipeline(ticker=ticker, keywords=keywords)
            
            if analysed_data is not None and not analysed_data.empty:
                run_model_training_pipeline(ticker=ticker)
                print(f"--> ✅ Successfully trained model for {ticker}")
            else:
                print(f"--> No news data for {ticker}. Skipping model training.")
        except Exception as e:
            print(f"--- ❌ FAILED to process {ticker}. Error: ---")
            traceback.print_exc()
            print("-------------------------------------------------")
            continue # Move to the next ticker

if __name__ == "__main__":
    main()