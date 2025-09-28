# backtest.py (Definitive Final Version)

import backtrader as bt
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from predict import get_prediction
from tqdm import tqdm

# --- 1. The Strategy Definition ---
class TFTStrategy(bt.Strategy):
    params = (
        ("tickers", []),
    )

    def __init__(self):
        # Add a timer to trigger our logic once a month
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=[1],
            monthcarry=True,
        )
        self.rebalance_targets = {}

    def get_expected_returns(self):
        """Generates forecasts and calculates expected returns."""
        expected_returns = {}
        print(f"\n--- [{self.datas[0].datetime.date(0)}] Generating new forecasts ---")
        for ticker in tqdm(self.p.tickers, desc="Forecasting"):
            prediction_data, final_df = get_prediction(ticker)
            if prediction_data is not None:
                current_price = self.getdatabyname(ticker).close[0]
                predictions = prediction_data.output[0].cpu().numpy()
                median_forecast_price = predictions[-1, 3]
                expected_return = (median_forecast_price - current_price) / current_price
                expected_returns[ticker] = expected_return
            else:
                expected_returns[ticker] = 0.0
        return pd.Series(expected_returns)

    def get_optimal_weights(self, expected_returns_vec):
        """Calculates optimal portfolio weights."""
        hist_prices = []
        for ticker in self.p.tickers:
            # Create a pandas series from the backtrader data line
            dates = self.getdatabyname(ticker).datetime.get(size=252)
            prices = self.getdatabyname(ticker).close.get(size=252)
            df = pd.Series(prices, index=pd.to_datetime(dates))
            hist_prices.append(df)

        prices_df = pd.concat(hist_prices, axis=1)
        prices_df.columns = self.p.tickers
        prices_df = prices_df.dropna()
        
        if prices_df.empty: return {}

        S = risk_models.sample_cov(prices_df)
        ef = EfficientFrontier(expected_returns_vec, S)
        try:
            weights = ef.max_sharpe()
            return ef.clean_weights()
        except Exception as e:
            print(f"Could not calculate optimal weights: {e}")
            return {}

    def notify_timer(self, timer, when, *args, **kwargs):
        """This function is called when the monthly timer triggers."""
        mu = self.get_expected_returns()
        self.rebalance_targets = self.get_optimal_weights(mu)
        print(f"New Target Weights: {self.rebalance_targets}")

    def next(self):
        """Called on every bar; places orders to rebalance."""
        if self.rebalance_targets:
            for ticker in self.p.tickers:
                weight = self.rebalance_targets.get(ticker, 0)
                self.order_target_percent(data=self.getdatabyname(ticker), target=weight)
            self.rebalance_targets = {}
            
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            self.log(f'ORDER EXECUTED: {order.info.name}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ORDER FAILED: {order.info.name}')

# --- 2. The Backtesting Setup ---
if __name__ == "__main__":
    
    tickers_to_backtest = [
        "TATAMOTORS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ITC.NS", 
        "LT.NS", "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS"
    ]
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TFTStrategy, tickers=tickers_to_backtest)
    
    # --- THIS IS THE FIX ---
    # We now use pandas to load and parse the data first, which is more robust.
    for ticker in tickers_to_backtest:
        data_path = f'data/{ticker}_historical_data.csv'
        
        # Load data with pandas
        dataframe = pd.read_csv(data_path)
        # Parse the 'Date' column correctly
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        # Set 'Date' as the index
        dataframe.set_index('Date', inplace=True)

        # Feed the clean pandas DataFrame into backtrader
        data = bt.feeds.PandasData(
            dataname=dataframe,
            fromdate=pd.to_datetime('2024-01-01'),
            todate=pd.to_datetime('2025-09-27')
        )
        cerebro.adddata(data, name=ticker)
    # --- END OF FIX ---

    start_portfolio_value = 1000000
    cerebro.broker.setcash(start_portfolio_value)
    cerebro.broker.setcommission(commission=0.001)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Add an analyzer to track our portfolio value over time
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    # Run the backtest
    results = cerebro.run()
    
    end_portfolio_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value:   {end_portfolio_value:.2f}')
    print(f'Total Return:          {((end_portfolio_value / start_portfolio_value) - 1) * 100:.2f}%')
    
    # Plot the results
    cerebro.plot(style='candlestick', iplot=False, savefig=True, figfilename='backtest_results.png')