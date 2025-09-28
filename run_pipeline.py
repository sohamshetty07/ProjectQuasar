# run_pipeline.py (Corrected with importable get_prepared_data)

import pandas as pd
import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pandas_ta as ta

from rss_scraper import get_news_from_rss
from intelligence_layer import analyze_news_with_llm
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

def prepare_stock_data(file_path, stock_name):
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
        print(f"An error occurred during data prep for {stock_name}: {e}")
        return None

def get_prepared_data(ticker: str):
    """
    This is the new, reusable function that loads all data and engineers all features for a given ticker.
    """
    stock_data_file = f"data/{ticker}_historical_data.csv"
    analysed_news_file = f"data/{ticker}_analysed_news.csv"

    stock_df = prepare_stock_data(stock_data_file, ticker)
    if stock_df is None: return None
    
    if not os.path.exists(analysed_news_file):
        print(f"Warning: No analysed news file for {ticker}. Proceeding with quantitative data only.")
        news_df = pd.DataFrame()
    else:
        news_df = pd.read_csv(analysed_news_file)

    # --- Feature Engineering ---
    stock_df.ta.rsi(length=14, append=True)
    stock_df.ta.macd(append=True)
    stock_df.fillna(0, inplace=True)
    stock_df.columns = [col.replace('.', '_') for col in stock_df.columns]

    final_indicator_cols = ['RSI_14', 'MACDh_12_26_9']
    base_cols = ['Date', 'Close', 'Volume', 'time_idx', 'group', 'series']
    stock_df = stock_df[base_cols + final_indicator_cols]

    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date')
        news_df['sentiment_ewma'] = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
        news_df['impact_ewma'] = news_df.groupby(news_df['date'].dt.date)['predicted_impact'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
        ewma_features = news_df.groupby(news_df['date'].dt.date).last()[['sentiment_ewma', 'impact_ewma']].reset_index()
        flow_features = news_df.groupby(news_df['date'].dt.date).agg(
            news_count=('sentiment_score', 'count'), sentiment_volatility=('sentiment_score', 'std')
        ).reset_index()
        daily_news_features = pd.merge(ewma_features, flow_features, on='date', how='outer')
        final_df = pd.merge(stock_df, daily_news_features, left_on=stock_df['Date'].dt.date, right_on='date', how='left')
    else:
        final_df = stock_df.copy()
        # Add empty news columns if no news data exists
        for col in ["sentiment_ewma", "impact_ewma", "news_count", "sentiment_volatility"]:
            final_df[col] = 0

    feature_cols_to_fill = ['sentiment_ewma', 'impact_ewma', 'news_count', 'sentiment_volatility']
    final_df[feature_cols_to_fill] = final_df[feature_cols_to_fill].fillna(0)
    final_df = final_df.drop(columns=['date', 'key_0'], errors='ignore')
    return final_df


class TFTLightningWrapper(pl.LightningModule):
    # ... (This class is unchanged)
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = self.model.loss
    def forward(self, x):
        return self.model(x)
    def step(self, batch, batch_idx):
        x, y = batch
        target = y[0]
        output = self(x)
        prediction = output.prediction
        loss = self.loss(prediction, target)
        return loss
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.model.hparams.learning_rate)

PREDICTION_HORIZON = 30
HISTORY_LOOKBACK = 90
BATCH_SIZE = 128
EPOCHS = 15

def run_feature_engineering_pipeline(ticker: str, keywords: list):
    # ... (This function is unchanged)
    analysed_news_file = f"data/{ticker}_analysed_news.csv"
    print(f"\n--- 🚀 Feature Engineering for {ticker} ---")
    if os.path.exists(analysed_news_file):
        print(f"--> Found existing analysed news file for {ticker}. Skipping analysis.")
        return pd.read_csv(analysed_news_file)
    raw_articles = get_news_from_rss()
    if not raw_articles: return None
    analysed_results, relevant_articles_for_df = [], []
    for article in tqdm(raw_articles, desc=f"Analysing News for {ticker}"):
        if any(keyword in article['headline'].lower() for keyword in keywords):
             analysed_results.append(analyze_news_with_llm(article))
             relevant_articles_for_df.append(article)
    if not analysed_results:
        print(f"\nNo relevant articles found for {ticker}.")
        return None
    analysed_df = pd.DataFrame(analysed_results)
    analysed_df['date'] = pd.to_datetime([a['date'] for a in relevant_articles_for_df]).date
    analysed_df.to_csv(analysed_news_file, index=False)
    print(f"--> Saved analysed news for {ticker}.")
    return analysed_df

def run_model_training_pipeline(ticker: str):
    print(f"\n--- 🧠 Model Training for {ticker} ---")
    
    # Use the new shared function to get data
    final_df = get_prepared_data(ticker)
    if final_df is None or final_df.empty:
        print(f"--> Could not prepare data for {ticker}. Skipping training.")
        return
        
    max_time_idx = final_df["time_idx"].max()
    training_cutoff = max_time_idx - PREDICTION_HORIZON

    indicator_cols_unknown = ["Close", "Volume", "RSI_14", "MACDh_12_26_9"]
    training_dataset = TimeSeriesDataSet(
        final_df[lambda x: x.time_idx <= training_cutoff], time_idx="time_idx", target="Close",
        group_ids=["group"], time_varying_known_reals=["sentiment_ewma", "impact_ewma", "news_count", "sentiment_volatility"],
        time_varying_unknown_reals=indicator_cols_unknown, max_encoder_length=HISTORY_LOOKBACK,
        max_prediction_length=PREDICTION_HORIZON, allow_missing_timesteps=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, final_df, predict=True, stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)
    
    raw_tft = TemporalFusionTransformer.from_dataset(
        training_dataset, learning_rate=0.004333308424585354, hidden_size=128,
        attention_head_size=4, dropout=0.18612403457115634, hidden_continuous_size=32
    )
    model = TFTLightningWrapper(raw_tft)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{ticker}", filename="best-model", monitor="val_loss", mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="cpu", gradient_clip_val=0.1, callbacks=[checkpoint_callback],
        enable_progress_bar=False, enable_model_summary=False, logger=False
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    pass