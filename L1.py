# Initial Commit

# lOAD THE DATASET AND INSPECT DATA

import pandas as pd 
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

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
