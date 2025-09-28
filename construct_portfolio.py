# construct_portfolio.py (Corrected for yfinance update)

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from pypfopt import EfficientFrontier, risk_models

# Import our new, reusable prediction function
from predict import get_prediction

def get_expected_returns(tickers: list):
    """
    Generates forecasts for a list of tickers and calculates their expected returns.
    """
    expected_returns_series = {}
    
    for ticker in tqdm(tickers, desc="Generating Forecasts"):
        prediction_data, final_df = get_prediction(ticker)
        
        if prediction_data is not None:
            current_price = final_df['Close'].iloc[-1]
            predictions = prediction_data.output[0].cpu().numpy()
            median_forecast_price = predictions[-1, 3] # Last day, 4th quantile (median)
            expected_return = (median_forecast_price - current_price) / current_price
            expected_returns_series[ticker] = expected_return
        else:
            expected_returns_series[ticker] = 0.0
            
    return pd.Series(expected_returns_series)

def construct_optimal_portfolio(tickers: list, expected_returns_vec: pd.Series):
    """
    Constructs an optimal portfolio using the Efficient Frontier method.
    """
    print("\n--- 💰 Constructing Optimal Portfolio ---")
    
    # 1. Download the full multi-level dataframe first.
    all_data = yf.download(tickers, period="1y")
    
    # --- THIS IS THE FIX ---
    # The new yfinance default provides the adjusted price in the 'Close' column.
    prices = all_data['Close']
    # --- END OF FIX ---

    # Drop any stocks that have all NaN data (e.g., if it's a new listing)
    prices = prices.dropna(axis='columns', how='all')
    
    # Ensure the assets in our price data and expected returns match up
    common_tickers = list(set(prices.columns) & set(expected_returns_vec.index))
    prices = prices[common_tickers]
    expected_returns_vec = expected_returns_vec[common_tickers]

    if prices.empty:
        print("Could not fetch valid price data for any of the tickers. Aborting.")
        return

    # 3. Calculate the covariance matrix (the risk model)
    S = risk_models.sample_cov(prices)
    
    # 4. Optimise for the maximal Sharpe Ratio
    ef = EfficientFrontier(expected_returns_vec, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    print("\nOptimal Portfolio Weights:")
    print(cleaned_weights)
    
    print("\nPortfolio Performance:")
    ef.portfolio_performance(verbose=True)

if __name__ == "__main__":
    
    from update_tickers import get_nifty50_tickers
    portfolio_tickers = get_nifty50_tickers()
    
    mu = get_expected_returns(portfolio_tickers)
    
    construct_optimal_portfolio(portfolio_tickers, mu)