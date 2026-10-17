#   MULTISTEP FORECASTING - WE PREDICT MORE THAN THE SINGLE NEXT DAY TEMPERATURE
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import torch
from L1 import model_rnn, model_gru, model_lstm

S = 7  # history window length
H = 7  # forecast horizon

DATA_PATH = "1_Daily_minimum_temps.csv"

df = pd.read_csv(DATA_PATH)

# Loading CSV 
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")  
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")   
df = df.set_index("Date").sort_index()

# --- Keep the daily calendar intact: reindex to complete daily grid ---
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_idx)
df.index.name = "Date"

# Splitting dataset 
split_date = pd.Timestamp("1989-01-01")  

df_train = df.loc[: split_date - pd.Timedelta(days=1)].copy()
df_test  = df.loc[split_date :].copy()

df_train["Temp"] = df_train["Temp"].interpolate(method="time", limit_direction="both")
df_test["Temp"]  = df_test["Temp"].interpolate(method="time",  limit_direction="both")

# SCALING INPUTS TO MAKE THE TRAINING SMOOTHER/FAVORABLE FOR THE MODEL
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[["Temp"]])                 # train-only fit

train_scaled = scaler.transform(df_train[["Temp"]])  #we are scaling only the training data to avoid data leakage
test_scaled  = scaler.transform(df_test[["Temp"]])

# A HELPER FUNCTION TO CREATE THE DATASET
def create_multi_step_sequences(values: np.ndarray, S: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D (or Nx1) array 'values' into (X, y) pairs for multi-step forecasting.

    Args:
      values: array-like of shape (N,) or (N,1) containing a single time series (scaled or raw).
      S: input window length (history steps).
      H: forecast horizon length (future steps).

    Returns:
      X: ndarray of shape (N_samples, S, 1)   [batch-first windows]
      y: ndarray of shape (N_samples, H)      [multi-step targets]

    Notes:
      - We build windows strictly within the given 'values' segment (no boundary crossing).
      - We keep the final singleton feature dimension so X is ready for batch-first RNN/LSTM/GRU.
    """
    v = values.reshape(-1)  # flatten to 1D array
    N = len(v)
    N_samples = N - (S + H) + 1
    if N_samples <= 0:
        raise ValueError(f"Not enough data points ({N}) for S={S}, H={H}. Need at least S+H samples.")
    X = np.zeros((N_samples, S, 1), dtype=np.float32)
    Y = np.zeros((N_samples, H),   dtype=np.float32)
    for i in range(N_samples):
        X[i, :, 0] = v[i : i + S] # 0 means 
        Y[i, :]    = v[i + S : i + S + H]
    return X, Y

X_train_sc, y_train_sc = create_multi_step_sequences(train_scaled, S=S, H=H)
X_test_sc,  y_test_sc  = create_multi_step_sequences(test_scaled,  S=S, H=H)

# Defining baseline persistene and baseline mean 
def baseline_persistence(X_windows: np.ndarray, H: int, ) -> np.ndarray:
    last_vals = X_windows[:,-1,0]
    return np.tile(last_vals.reshape(-1,1)), (1, H)

def baseline_mean_S(X_windows: np.ndarray, H: int):
    means = X_windows[:,:,0].mean(axis=1)
    return np.tile(means.reshape(-1,1), (1,H))

for m in (model_rnn, model_lstm, model_gru):
    m.eval()

USE_CLAMP = False

@torch.no_grad()
def rollout_recursive(model, X_windows_scaled, H, clamp=(0.0, 1.0)):
    """
    Args:
      model: trained single-step model that maps (B,S,1) -> (B,1) in scaled space.
      X_windows_scaled: numpy array (N, S, 1) in scaled space for the starting windows.
      H: forecast horizon (number of steps to roll out).
      clamp: tuple (min, max) to clamp predictions if USE_CLAMP is True.
    Returns:
      y_pred_scaled: numpy array (N, H) of scaled forecasts.
    """
    model.eval()
    curr = torch.tensor(X_windows_scaled, dtype=torch.float32)
    N = curr.size(0)
    preds = torch.zeros((N, H), dtype=torch.float32)
    for k in range(H):
        y_hat = model(curr)
        if USE_CLAMP:
            y_hat = y_hat.clamp(min=clamp[0], max=clamp[1])
        preds[:, k] = y_hat.squeeze(1)
        curr = torch.cat([curr[:, 1:, :], y_hat.unsqueeze(-1)], dim=1)
    return preds.numpy()

# Rollout recursive function is going to rollout the predictions for the next H days

# 1) Roll out (scaled)
yhat_rnn_sc  = rollout_recursive(model_rnn,  X_test_sc, H) 
yhat_lstm_sc = rollout_recursive(model_lstm, X_test_sc, H) 
yhat_gru_sc  = rollout_recursive(model_gru,  X_test_sc, H) 

# 2) Inverse-transform to °C for evaluation/plots
def inverse_scale_2d(y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform (N,H) using the same scaler trained on train; returns (N,H) in °C."""
    flat = y_scaled.reshape(-1, 1)
    back = scaler.inverse_transform(flat)
    return back.reshape(y_scaled.shape)

yhat_rnn_C  = inverse_scale_2d(yhat_rnn_sc)   if yhat_rnn_sc  is not None else None
yhat_lstm_C = inverse_scale_2d(yhat_lstm_sc)  if yhat_lstm_sc is not None else None
yhat_gru_C  = inverse_scale_2d(yhat_gru_sc)   if yhat_gru_sc  is not None else None

# Direct Multi-output head 
# Instead of rollout recursive method we try to predic the whole horizons once 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Lets reload the dataset
BATCH_SIZE = 64

train_ds_direct = TensorDataset(
    torch.tensor(X_train_sc, dtype=torch.float32),
    torch.tensor(y_train_sc, dtype=torch.float32)
)
test_ds_direct = TensorDataset(
    torch.tensor(X_test_sc, dtype=torch.float32),
    torch.tensor(y_test_sc, dtype=torch.float32)
)

train_loader_direct = DataLoader(train_ds_direct, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader_direct = DataLoader(test_ds_direct, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# The GRU model that predicts all at once model
class GRUModelDirect(nn.Module):
    def __init__(self, input_size=1,hidden_size= 64, num_layers=1, H=7, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.head = nn.Linear(hidden_size, H)

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeroz(self.gru.num_layers, B, self.gru.hidden_size)
        out, _ = self.gru(x, h0)
        h_last = out[:,-1, :]
        y_hat = self.head(h_last)
        return y_hat

# Instantiate the model
H = y_train_sc.shape[1] #ensure we use the same horizon as our data
GRU_model_direct = GRUModelDirect(input_size=1, hidden_size=64, num_layers=1, H=H, batch_first=True)

