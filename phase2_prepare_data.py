import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# --- Configuration ---
DATA_FILE = "data/TATAMOTORS.NS_historical_data.csv"
STOCK_NAME = "TATAMOTORS.NS"
PREDICTION_HORIZON = 30
HISTORY_LOOKBACK = 90

# --- Main Script ---

def prepare_stock_data(file_path, stock_name):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Volume'] = df['Volume'].astype(float)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
        df['group'] = stock_name
        df['series'] = 0
        df_processed = df[['Date', 'Close', 'Volume', 'time_idx', 'group', 'series']]
        return df_processed
    except Exception as e:
        print(f"An error occurred during data prep: {e}")
        return None

if __name__ == "__main__":
    prepared_df = prepare_stock_data(DATA_FILE, STOCK_NAME)
    
    if prepared_df is not None:
        print("\n--- Creating the TimeSeriesDataSet ---")
        
        max_time_idx = prepared_df["time_idx"].max()
        training_cutoff = max_time_idx - PREDICTION_HORIZON
        
        training_dataset = TimeSeriesDataSet(
            prepared_df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="Close",
            group_ids=["group"],
            time_varying_unknown_reals=["Close", "Volume"],
            max_encoder_length=HISTORY_LOOKBACK,
            max_prediction_length=PREDICTION_HORIZON,
            # --- THIS IS THE FIX ---
            # Allow for missing time steps (weekends, holidays)
            allow_missing_timesteps=True,
            # --- END OF FIX ---
            scalers={
                "Volume": GroupNormalizer(transformation="log")
            }
        )
        
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, prepared_df, predict=True, stop_randomization=True
        )

        print("\n--- Dataset Summary ---")
        print(training_dataset)
        
        print(f"\nSuccessfully created training and validation datasets.")
        
        print("\nNext step: Create data loaders and build the TFT model.")