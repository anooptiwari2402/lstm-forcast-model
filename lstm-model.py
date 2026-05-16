import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import ta
import matplotlib.pyplot as plt

#############################################
# CONFIGURATION
#############################################

CSV_FILE = "stock_data.csv"

SEQUENCE_LENGTH = 60
FORECAST_DAYS = 30

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# LOAD CSV
#############################################

df = pd.read_csv(CSV_FILE)

#############################################
# CLEANING
#############################################

df.columns = [c.strip() for c in df.columns]

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values('Date', inplace=True)

df.reset_index(drop=True, inplace=True)

#############################################
# FEATURE ENGINEERING
#############################################

# RSI
df['RSI'] = ta.momentum.RSIIndicator(
    close=df['Close Price'],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(close=df['Close Price'])

df['MACD'] = macd.macd()

# EMA
df['EMA20'] = ta.trend.EMAIndicator(
    close=df['Close Price'],
    window=20
).ema_indicator()

# Bollinger Bands
bb = ta.volatility.BollingerBands(
    close=df['Close Price'],
    window=20
)

df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()

#############################################
# REMOVE NULLS
#############################################

df.dropna(inplace=True)

#############################################
# FEATURES
#############################################

FEATURE_COLUMNS = [
    'Open Price',
    'High Price',
    'Low Price',
    'Close Price',
    'WAP',
    'No.of Shares',
    'No. of Trades',
    'Total Turnover (Rs.)',
    'Deliverable Quantity',
    '% Deli. Qty to Traded Qty',
    'Spread High-Low',
    'Spread Close-Open',
    'RSI',
    'MACD',
    'EMA20',
    'BB_High',
    'BB_Low'
]

TARGET_COLUMN = 'Close Price'

#############################################
# SCALING
#############################################

feature_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(
    df[FEATURE_COLUMNS]
)

target_scaler = MinMaxScaler()

scaled_target = target_scaler.fit_transform(
    df[[TARGET_COLUMN]]
)

#############################################
# CREATE SEQUENCES
#############################################

X = []
y = []

for i in range(SEQUENCE_LENGTH, len(df)):

    X.append(
        scaled_features[i-SEQUENCE_LENGTH:i]
    )

    y.append(
        scaled_target[i]
    )

X = np.array(X)
y = np.array(y)

#############################################
# TRAIN TEST SPLIT
#############################################

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

#############################################
# DATASET
#############################################

class StockDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#############################################
# LSTM MODEL
#############################################

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(128, 64)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

model = LSTMModel(
    input_size=len(FEATURE_COLUMNS)
).to(DEVICE)

#############################################
# LOSS & OPTIMIZER
#############################################

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#############################################
# TRAINING
#############################################

print("Training Started...")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.6f}"
    )

print("Training Completed.")

#############################################
# FORECASTING NEXT 30 DAYS
#############################################

model.eval()

forecast_input = scaled_features[-SEQUENCE_LENGTH:]

forecast_input = np.expand_dims(
    forecast_input,
    axis=0
)

forecast_days = []

for day in range(FORECAST_DAYS):

    input_tensor = torch.tensor(
        forecast_input,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        prediction = model(input_tensor)

    predicted_price_scaled = prediction.cpu().numpy()

    predicted_price = target_scaler.inverse_transform(
        predicted_price_scaled
    )[0][0]

    forecast_days.append(predicted_price)

    #########################################
    # CREATE NEXT INPUT
    #########################################

    new_row = forecast_input[0][-1].copy()

    close_index = FEATURE_COLUMNS.index('Close Price')

    new_row[close_index] = predicted_price_scaled[0][0]

    forecast_input = np.append(
        forecast_input[:, 1:, :],
        [[new_row]],
        axis=1
    )

#############################################
# OUTPUT FORECAST
#############################################

forecast_df = pd.DataFrame({
    "Day": range(1, FORECAST_DAYS + 1),
    "Forecast_Close_Price": forecast_days
})

print("\n")
print("30 Day Forecast")
print(forecast_df)

#############################################
# SAVE FORECAST
#############################################

forecast_df.to_csv(
    "forecast_output.csv",
    index=False
)

#############################################
# PLOT
#############################################

plt.figure(figsize=(14, 7))

plt.plot(
    forecast_days,
    label='Forecasted Price'
)

plt.title("30 Day Stock Forecast")

plt.xlabel("Days")

plt.ylabel("Price")

plt.legend()

plt.show()