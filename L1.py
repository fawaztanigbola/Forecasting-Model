# Initial Commit

# lOAD THE DATASET AND INSPECT DATA

import pandas as pd 
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

# Load raw csv 
df = pd.read_csv("1_Daily_minimum_temps.csv")

# Parse Date (format is MM, DD, YY, e.g , "01/13/81")
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")

# Set date as index 
df.set_index("Date", inplace=True)

# Ensure temperature is numeric
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

#--------------------------------
#   CREATING SLIDING WINDOWS 
# -------------------------------

def create_sliding_windows(values, window_size=3):
    """
    Convert a 1D array ino (X,y) pairs using a sliding window
    X  shape: (num_samples, window_size)
    y_shape: (num_samples,)
    """
    x, y = [], []
    for i in range(window_size, len(values)):
        x.append(values[i-window_size:i])
        y.append(values[i])
        return np.array(x), np.array(y)

# SLIDE THE WINDOWS  ACROSS DATASET
temps = df["temp"].values

x_real, y_real = create_sliding_windows(temps, window_size=7)

# SPLITTING DATASET BEFORE CREATING WINDOWS TO AVOID DATA LEAKAGE
# Splitting data into 8 years for training (1981-1988)
# and two years for testing(1989-1990)

split_date = pd.Timestamp("1989-01-01")

df_train = df.loc[:split_date- pd.Timedelta(days=1)]
df_test = df.loc[split_date]

# LOADING DATA SEPARATELY(TEST - TRAIN)
window_size  = 7

X_train, y_train = create_sliding_windows(df_train["temp"].values, window_size=window_size)
X_test, y_test = create_sliding_windows(df_test["temp"].values, window_size=window_size)

# SCALING THE TEMPERATURE VAUES 

# Initialize Scaler 
scaler = MinMaxScaler(feature_range=(0,1))

# Fit only on training set
scaler.fit(df_train[["Temp"]])

# Transform train and test
train_scaled = scaler.transform(df_train[["Temp"]])
test_scaled = scaler.transform(df_test[["Temp"]])

# CREATING SLIDING WINDOWS ON THE SCALED DATA
X_train, y_train = create_sliding_windows(train_scaled, window_size=window_size)
X_test, y_test = create_sliding_windows(test_scaled, window_size=window_size)

# PREPARING PYTORCH DATASETS AND LOADERS

# convert to tensors 
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Lo\ad data using dataloader
BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=True)


# An LSTM is just an RNN with gates
# 

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=batch_first,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        X: (B, S, F) because we use batch_first = True 
        returns: (B, 1)
        """

        B = x.size(0)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        out, hnn = self.rnn(x, h0)

        h_last = out[:,-1,:] 

        y_hat = self.head(h_last)
        return y_hat

# LSTM Model

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (B, S, F) when batch_first=True (e.g., (B, 7, 1))
        returns: (B, 1)
        """
        B = x.size(0)
        
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))   

        h_last = out[:, -1, :]                 

        y_hat = self.head(h_last)                
        return y_hat

# GATED RECURRENT UNIT (GRU) - IT IS LIGHTER THAN LSTM WITH SIMILAR RESULT

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False
        )
        self.head = nn.Linear(hidden_size, 1)  

    def forward(self, x):
        """
        x: (B, S, F) because we use batch_first=True (e.g., (B, 7, 1))
        returns: (B, 1)
        """
        B = x.size(0)

        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        out, hn = self.gru(x, h0)

        h_last = out[:, -1, :]     

        y_hat = self.head(h_last)  
        return y_hat

# TRAINING LOOP FOR ANY MODEL(RNN/ LSTM/ GRU)
def train_one_model(model, train_loader, val_loader, epochs=20, lr=1e-3, clip=None, verbose=True):
    """
    Train a sequence model (RNN/LSTM/GRU head) on scaled targets.
    Assumes:
      - inputs:  (B, S, F) with batch_first=True (as built in our loaders)
      - targets: (B, 1), scaled to [0,1]
    Returns:
      model (trained in-place), history: {'train_loss': [...], 'val_loss': [...]}
    """
    device = torch.device("cpu")  
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        #  Train 
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()

            if clip is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        train_loss = running / max(n, 1)
        history["train_loss"].append(train_loss)

        #  Validate 
        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                y_hat = model(xb)
                loss = criterion(y_hat, yb)
                running += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = running / max(n, 1)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

    return model, history

#   RNNModel, LSTMModel, GRUModel

EPOCHS = 20
LR     = 1e-3
CLIP   = 1.0

# RNN baseline
model_rnn = RNNModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_rnn, hist_rnn = train_one_model(model_rnn, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)

# LSTM
model_lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_lstm, hist_lstm = train_one_model(model_lstm, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)

# GRU
model_gru = GRUModel(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
model_gru, hist_gru = train_one_model(model_gru, train_loader, test_loader, epochs=EPOCHS, lr=LR, clip=CLIP)

# Our predictions are scaled we have to convert them back to degree celsius
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_on_loader(model, loader):
    """
    Returns:
      y_true_scaled: (N, 1)
      y_pred_scaled: (N, 1)
    """
    model.eval()
    ys, yhs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yhat = model(xb)
            ys.append(yb.cpu().numpy())
            yhs.append(yhat.cpu().numpy())
    y_true_scaled = np.vstack(ys)
    y_pred_scaled = np.vstack(yhs)
    return y_true_scaled, y_pred_scaled


def inverse_to_celsius(arr_scaled, scaler):
    """arr_scaled: (N,1) → inverse_transform → (N,1) in °C"""
    return scaler.inverse_transform(arr_scaled)


def evaluate_in_celsius(model, loader, scaler):
    """
    Returns:
      metrics dict and (y_true_C, y_pred_C) arrays, each (N,1)
    """
    y_s, yhat_s = predict_on_loader(model, loader)
    y_C    = inverse_to_celsius(y_s,    scaler)
    yhat_C = inverse_to_celsius(yhat_s, scaler)

    mae  = mean_absolute_error(y_C, yhat_C)
    mse  = mean_squared_error(y_C, yhat_C)
    rmse = mse ** 0.5
    r2   = r2_score(y_C, yhat_C)

    metrics = {"MAE_C": float(mae), "RMSE_C": float(rmse), "R2": float(r2)}
    return metrics, y_C, yhat_C

# EVALUATING MODELS
# RNN
metrics_rnn, yC_rnn, yhC_rnn = evaluate_in_celsius(model_rnn, test_loader, scaler)
print("RNN:", metrics_rnn)

# LSTM
metrics_lstm, yC_lstm, yhC_lstm = evaluate_in_celsius(model_lstm, test_loader, scaler)
print("LSTM:", metrics_lstm)

# GRU
metrics_gru, yC_gru, yhC_gru = evaluate_in_celsius(model_gru, test_loader, scaler)
print("GRU:", metrics_gru)
