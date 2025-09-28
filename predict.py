# predict.py (Updated for Reusability)

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import glob
import os
import argparse
import pandas as pd

from run_pipeline import TFTLightningWrapper, TimeSeriesDataSet, TemporalFusionTransformer, get_prepared_data

def get_prediction(ticker: str):
    """
    Runs the prediction pipeline and returns the raw forecast data.
    """
    checkpoint_dir = f"checkpoints/{ticker}"
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not list_of_files:
        print(f"Error: No checkpoint file found for {ticker}.")
        return None, None
    latest_checkpoint = max(list_of_files, key=os.path.getctime)

    final_df = get_prepared_data(ticker)
    if final_df is None or final_df.empty:
        return None, None

    PREDICTION_HORIZON = 30
    HISTORY_LOOKBACK = 90
    max_time_idx = final_df["time_idx"].max()
    training_cutoff = max_time_idx - PREDICTION_HORIZON
    
    indicator_cols_unknown = ["Close", "Volume", "RSI_14", "MACDh_12_26_9"]
    training_dataset = TimeSeriesDataSet(
        final_df[lambda x: x.time_idx <= training_cutoff], time_idx="time_idx", target="Close",
        group_ids=["group"], time_varying_known_reals=["sentiment_ewma", "impact_ewma", "news_count", "sentiment_volatility"],
        time_varying_unknown_reals=indicator_cols_unknown, max_encoder_length=HISTORY_LOOKBACK,
        max_prediction_length=PREDICTION_HORIZON, allow_missing_timesteps=True,
    )

    raw_model = TemporalFusionTransformer.from_dataset(
        training_dataset, learning_rate=0.0043, hidden_size=128, attention_head_size=4,
        dropout=0.1861, hidden_continuous_size=32
    )
    best_model_wrapper = TFTLightningWrapper.load_from_checkpoint(latest_checkpoint, model=raw_model)
    
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, final_df, predict=True, stop_randomization=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
    
    raw_predictions = best_model_wrapper.model.predict(
        val_dataloader, mode="quantiles", return_x=True, trainer_kwargs={"accelerator": "cpu"}
    )
    return raw_predictions, final_df

def run_prediction_plot(ticker: str):
    print(f"--- 🚀 Starting Prediction Pipeline for {ticker} ---")
    raw_predictions, final_df = get_prediction(ticker)
    
    if raw_predictions is None:
        print(f"Could not generate prediction for {ticker}.")
        return

    print("--> Plotting prediction...")
    x, y_tuple = raw_predictions.x, raw_predictions.output
    history = x['encoder_target'][0].cpu().numpy()
    true_future = x['decoder_target'][0].cpu().numpy()
    predictions = y_tuple[0].cpu().numpy()
    
    median_prediction = predictions[:, 3]
    lower_bound = predictions[:, 1]
    upper_bound = predictions[:, 5]
    
    history_index = final_df["Date"].iloc[-len(history)-len(true_future) : -len(true_future)]
    future_index = final_df["Date"].iloc[-len(true_future):]
    
    plt.figure(figsize=(12, 7))
    plt.plot(history_index, history, label="Historical Data")
    plt.plot(future_index, true_future, label="True Future", color='orange')
    plt.plot(future_index, median_prediction, label="Median Prediction", color='blue', linestyle='--')
    plt.fill_between(future_index, lower_bound, upper_bound, alpha=0.3, label="Prediction Interval (10%-90%)", color='cyan')
    
    plt.title(f"Prediction for {ticker} | 30-day forecast")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f"prediction_plot_{ticker}.png"
    plt.savefig(plot_filename)
    print(f"\n--- ✅ SUCCESS! Plot saved to {plot_filename} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a forecast for a given stock ticker.")
    parser.add_argument("ticker", type=str, help="The stock ticker to predict (e.g., TATAMOTORS.NS)")
    args = parser.parse_args()
    run_prediction_plot(ticker=args.ticker)