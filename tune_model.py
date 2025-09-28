# tune_model.py (Complete Corrected Version 5)

import optuna
import torch
import pytorch_lightning as pl
import pandas as pd
import pandas_ta as ta

# --- Important: Import the necessary components from your run_pipeline script ---
from run_pipeline import (
    TFTLightningWrapper,
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

# --- Configuration (mirrors run_pipeline.py) ---
STOCK_TICKER = "TATAMOTORS.NS"
STOCK_DATA_FILE = f"data/{STOCK_TICKER}_historical_data.csv"
ANALYSED_NEWS_FILE = f"data/{STOCK_TICKER}_analysed_news.csv"
PREDICTION_HORIZON = 30
HISTORY_LOOKBACK = 90
BATCH_SIZE = 128


def prepare_stock_data(file_path, stock_name):
    """
    Loads and does initial prep of the stock CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Volume'] = df['Volume'].astype(float)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
        df['group'] = stock_name
        df['series'] = 0
        df_processed = df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'time_idx', 'group', 'series']]
        return df_processed
    except Exception as e:
        print(f"An error occurred during data prep: {e}")
        return None


def get_prepared_data():
    """
    This function encapsulates all the data loading and feature engineering.
    """
    stock_df = prepare_stock_data(STOCK_DATA_FILE, STOCK_TICKER)
    news_df = pd.read_csv(ANALYSED_NEWS_FILE)

    # 1. Engineer technical indicators
    stock_df.ta.rsi(length=14, append=True)
    stock_df.ta.macd(append=True)
    # --- FIX: Temporarily removing Bollinger Bands to resolve the stubborn KeyError ---
    # stock_df.ta.bbands(length=20, append=True)
    stock_df.fillna(0, inplace=True)
    stock_df.columns = [col.replace('.', '_') for col in stock_df.columns]

    # Explicitly select only the columns we intend to use.
    # --- FIX: Removed 'BBP_20_2_0' from this list ---
    final_indicator_cols = ['RSI_14', 'MACDh_12_26_9']
    base_cols = ['Date', 'Close', 'Volume', 'time_idx', 'group', 'series']
    stock_df = stock_df[base_cols + final_indicator_cols]

    # 2. Engineer advanced news features
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.sort_values(by='date')
    news_df['sentiment_ewma'] = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    news_df['impact_ewma'] = news_df.groupby(news_df['date'].dt.date)['predicted_impact'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    ewma_features = news_df.groupby(news_df['date'].dt.date).last()[['sentiment_ewma', 'impact_ewma']].reset_index()
    flow_features = news_df.groupby(news_df['date'].dt.date).agg(
        news_count=('sentiment_score', 'count'),
        sentiment_volatility=('sentiment_score', 'std')
    ).reset_index()
    daily_news_features = pd.merge(ewma_features, flow_features, on='date', how='outer')

    # 3. Merge all features
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    final_df = pd.merge(stock_df, daily_news_features, left_on=stock_df['Date'].dt.date, right_on='date', how='left')
    feature_cols_to_fill = ['sentiment_ewma', 'impact_ewma', 'news_count', 'sentiment_volatility']
    final_df[feature_cols_to_fill] = final_df[feature_cols_to_fill].fillna(0)
    final_df = final_df.drop(columns=['date', 'key_0'], errors='ignore')

    return final_df


def objective(trial: optuna.Trial):
    """
    This is the main function for Optuna. It trains a model with a given
    set of hyperparameters and returns its validation loss.
    """
    pl.seed_everything(42)

    # 1. Suggest hyperparameters for this trial
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 4])
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32])

    # 2. Setup datasets and dataloaders
    final_df = get_prepared_data()
    max_time_idx = final_df["time_idx"].max()
    training_cutoff = max_time_idx - PREDICTION_HORIZON
    
    # --- FIX: Removed 'BBP_20_2_0' from this list ---
    indicator_cols_unknown = ["Close", "Volume", "RSI_14", "MACDh_12_26_9"]
    
    training_dataset = TimeSeriesDataSet(
        final_df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx", target="Close", group_ids=["group"],
        time_varying_known_reals=["sentiment_ewma", "impact_ewma", "news_count", "sentiment_volatility"],
        time_varying_unknown_reals=indicator_cols_unknown,
        max_encoder_length=HISTORY_LOOKBACK, max_prediction_length=PREDICTION_HORIZON,
        allow_missing_timesteps=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, final_df, predict=True, stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)

    # 3. Create model with suggested params
    raw_tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
    )
    model = TFTLightningWrapper(raw_tft)

    # 4. Setup trainer and run training
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="cpu",
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 5. Return the final validation loss
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    print("--- 🧠 Starting Hyperparameter Optimisation with Optuna ---")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n--- ✅ Optimisation Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (val_loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")