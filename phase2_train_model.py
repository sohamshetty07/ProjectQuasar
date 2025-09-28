import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.utils import move_to_device

# --- Configuration ---
DATA_FILE = "data/TATAMOTORS.NS_historical_data.csv"
STOCK_NAME = "TATAMOTORS.NS"
PREDICTION_HORIZON = 30
HISTORY_LOOKBACK = 90
BATCH_SIZE = 128
EPOCHS = 10

# --- 1. Data Preparation Function (Same as before) ---
def prepare_stock_data(file_path, stock_name):
    # This function is unchanged
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

# --- 2. Corrected LightningModule Wrapper ---
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = self.model.loss

    def forward(self, x):
        # --- THIS IS THE FIX ---
        # Manually move the input data to the model's device (e.g., 'mps')
        x = move_to_device(x, self.device)
        # --- END OF FIX ---
        return self.model(x)

    def step(self, batch, batch_idx):
        x, y = batch
        target = y[0]
        output = self(x)
        prediction = output.prediction
        
        # The target also needs to be on the correct device for the loss calculation
        loss = self.loss(prediction, move_to_device(target, self.device))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.model.hparams.learning_rate)

# --- Main Script ---
if __name__ == "__main__":
    pl.seed_everything(42)
    prepared_df = prepare_stock_data(DATA_FILE, STOCK_NAME)
    
    if prepared_df is not None:
        max_time_idx = prepared_df["time_idx"].max()
        training_cutoff = max_time_idx - PREDICTION_HORIZON
        
        training_dataset = TimeSeriesDataSet(
            prepared_df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx", target="Close", group_ids=["group"],
            time_varying_unknown_reals=["Close", "Volume"],
            max_encoder_length=HISTORY_LOOKBACK, max_prediction_length=PREDICTION_HORIZON,
            allow_missing_timesteps=True,
            scalers={"Volume": GroupNormalizer(transformation="log")}
        )
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, prepared_df, predict=True, stop_randomization=True
        )

        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)

        raw_tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
        )
        
        model = TFTLightningWrapper(raw_tft)

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="cpu",
            gradient_clip_val=0.1,
        )

        print("\nStarting model training...")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        print("Training complete.")