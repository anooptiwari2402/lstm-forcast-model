#########################################################
# INSTALL REQUIRED LIBRARIES
#########################################################

#########################################################
# IMPORTS
#########################################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import ta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.data import GroupNormalizer

from pytorch_forecasting.metrics import RMSE

from lightning.pytorch import Trainer

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor
)

#########################################################
# CONFIG
#########################################################

CSV_FILE = "/content/stock_data.csv"

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 30

BATCH_SIZE = 64
EPOCHS = 30

#########################################################
# LOAD CSV
#########################################################

df = pd.read_csv(CSV_FILE)

#########################################################
# CLEAN COLUMN NAMES PROPERLY
#########################################################

df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

print("Columns:")
print(df.columns.tolist())

#########################################################
# DATE COLUMN
#########################################################

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values("Date", inplace=True)

df.reset_index(drop=True, inplace=True)

#########################################################
# NUMERIC COLUMNS
#########################################################

numeric_cols = [
    'Open_Price',
    'High_Price',
    'Low_Price',
    'Close_Price',
    'WAP',
    'No_of_Shares',
    'No_of_Trades',
    'Total_Turnover_Rs',
    'Deliverable_Quantity',
    'Deli_Qty_to_Traded_Qty',
    'Spread_High_Low',
    'Spread_Close_Open'
]

#########################################################
# CONVERT TO NUMERIC
#########################################################

for col in numeric_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

#########################################################
# REMOVE NULLS
#########################################################

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

#########################################################
# FEATURE ENGINEERING
#########################################################

# RSI
df['RSI'] = ta.momentum.RSIIndicator(
    close=df['Close_Price'],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(
    close=df['Close_Price']
)

df['MACD'] = macd.macd()

# EMA20
df['EMA20'] = ta.trend.EMAIndicator(
    close=df['Close_Price'],
    window=20
).ema_indicator()

# Bollinger Bands
bb = ta.volatility.BollingerBands(
    close=df['Close_Price'],
    window=20
)

df['BB_High'] = bb.bollinger_hband()

df['BB_Low'] = bb.bollinger_lband()

#########################################################
# REMOVE NAN CREATED BY INDICATORS
#########################################################

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

#########################################################
# TIME INDEX
#########################################################

df["time_idx"] = np.arange(len(df))

#########################################################
# GROUP ID
#########################################################

df["series"] = "BSE_STOCK"

#########################################################
# TARGET
#########################################################

TARGET = "Close_Price"

#########################################################
# FEATURE COLUMNS
#########################################################

feature_cols = [
    'Open_Price',
    'High_Price',
    'Low_Price',
    'WAP',
    'No_of_Shares',
    'No_of_Trades',
    'Total_Turnover_Rs',
    'Deliverable_Quantity',
    'Deli_Qty_to_Traded_Qty',
    'Spread_High_Low',
    'Spread_Close_Open',
    'RSI',
    'MACD',
    'EMA20',
    'BB_High',
    'BB_Low'
]

#########################################################
# SCALE FEATURES
# DO NOT SCALE TARGET
#########################################################

scaler = MinMaxScaler()

df[feature_cols] = scaler.fit_transform(
    df[feature_cols]
)

#########################################################
# TRAIN / VALIDATION SPLIT
#########################################################

training_cutoff = (
    df["time_idx"].max()
    - MAX_PREDICTION_LENGTH
)

#########################################################
# DATASET
#########################################################

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],

    time_idx="time_idx",

    target=TARGET,

    group_ids=["series"],

    max_encoder_length=MAX_ENCODER_LENGTH,

    max_prediction_length=MAX_PREDICTION_LENGTH,

    static_categoricals=["series"],

    time_varying_known_reals=[
        "time_idx"
    ],

    time_varying_unknown_reals=[
        TARGET
    ] + feature_cols,

    target_normalizer=GroupNormalizer(
        groups=["series"]
    ),

    add_relative_time_idx=True,

    add_target_scales=True,

    add_encoder_length=True
)

#########################################################
# VALIDATION DATASET
#########################################################

validation = TimeSeriesDataSet.from_dataset(
    training,
    df,
    predict=True,
    stop_randomization=True
)

#########################################################
# DATALOADERS
#########################################################

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=0
)

val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0
)

#########################################################
# CALLBACKS
#########################################################

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5,
    verbose=True,
    mode="min"
)

lr_logger = LearningRateMonitor()

#########################################################
# TRAINER
#########################################################

trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[
        lr_logger,
        early_stop_callback
    ]
)

#########################################################
# TFT MODEL
#########################################################

tft = TemporalFusionTransformer.from_dataset(
    training,

    learning_rate=0.001,

    hidden_size=32,

    attention_head_size=4,

    dropout=0.1,

    hidden_continuous_size=16,

    output_size=1,

    loss=RMSE(),

    reduce_on_plateau_patience=3
)

#########################################################
# TRAIN MODEL
#########################################################

print("\nTraining TFT Model...\n")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

#########################################################
# LOAD BEST MODEL
#########################################################

best_model_path = trainer.checkpoint_callback.best_model_path

print("\nBest Model Path:")
print(best_model_path)

best_tft = TemporalFusionTransformer.load_from_checkpoint(
    best_model_path
)

#########################################################
# PREDICT
#########################################################

predictions = best_tft.predict(
    val_dataloader
)

#########################################################
# FORECAST VALUES
#########################################################

forecast = predictions.numpy().flatten()

#########################################################
# FORECAST DATAFRAME
#########################################################

forecast_df = pd.DataFrame({
    "Forecast_Day": range(1, len(forecast) + 1),
    "Forecast_Close_Price": forecast
})

#########################################################
# PRINT FORECAST
#########################################################

print("\n30 Day Forecast:\n")

print(forecast_df)

#########################################################
# SAVE CSV
#########################################################

forecast_df.to_csv(
    "/content/tft_forecast_output.csv",
    index=False
)

print("\nForecast saved to:")
print("/content/tft_forecast_output.csv")

#########################################################
# PLOT FORECAST
#########################################################

plt.figure(figsize=(14, 7))

plt.plot(
    forecast,
    label="TFT Forecast"
)

plt.title("30 Day Stock Forecast")

plt.xlabel("Days")

plt.ylabel("Predicted Close Price")

plt.legend()

plt.grid(True)

plt.show()