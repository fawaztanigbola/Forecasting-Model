# --- Code: Load CSV, strict date parsing, set Date index ---
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CSV_PATH = "1_Daily_minimum_temps.csv"

# Load as strings first
df_raw = pd.read_csv(CSV_PATH, dtype={"Date": str})
assert "Date" in df_raw.columns and "Temp" in df_raw.columns, "Expected columns: Date, Temp"

def try_parse_date(series, fmt):
    dt = pd.to_datetime(series, format=fmt, errors="coerce")
    return dt, dt.notna().sum()

# Try mm/dd/yy first, then dd/mm/yy; pick the one with fewer NaT
d1, ok1 = try_parse_date(df_raw["Date"], "%m/%d/%y")
d2, ok2 = try_parse_date(df_raw["Date"], "%d/%m/%y")

if ok1 >= ok2:
    df_raw["Date"] = d1
    chosen_fmt = "%m/%d/%y"
else:
    df_raw["Date"] = d2
    chosen_fmt = "%d/%m/%y"

print(f"Chosen date format: {chosen_fmt}  | parsed ok: {max(ok1, ok2)} / {len(df_raw)}")

# Drop unparsed rows (if any), cast Temp to float
df = df_raw.dropna(subset=["Date"]).copy()
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")

# Set index, sort, keep only the signal
df = df.set_index("Date").sort_index()[["Temp"]]

# Basic sanity
assert df.index.is_monotonic_increasing, "Date index must be sorted ascending"

# --- Code: Clean/impute tiny gaps safely ---
# Replace inf with NaN, then linear interpolation; fill edges if needed
s = df["Temp"].replace([np.inf, -np.inf], np.nan)
nan_before = s.isna().sum()

s = s.interpolate(method="linear", limit_direction="both")
nan_after = s.isna().sum()

df["Temp"] = s
print(f"Missing values before: {nan_before} | after interpolation: {nan_after}")

# --- Code: Time-aware split: Train (81-87), Val (88), Test (89-90) ---
train = df.loc["1981-01-01":"1987-12-31"].copy()
val   = df.loc["1988-01-01":"1988-12-31"].copy()
test  = df.loc["1989-01-01":"1990-12-31"].copy()

# Scaling the train dataset only and transforming the val/est data
# Numpy arrays (N,1)
train_np = train[["Temp"]].to_numpy(dtype=float)
val_np   = val[["Temp"]].to_numpy(dtype=float)
test_np  = test[["Temp"]].to_numpy(dtype=float)

scaler = MinMaxScaler()
train_sc = scaler.fit_transform(train_np)
val_sc   = scaler.transform(val_np)
test_sc  = scaler.transform(test_np)

print("Scaled ranges:")
print("  Train:", float(train_sc.min()), "→", float(train_sc.max()))
print("  Val  :", float(val_sc.min()),   "→", float(val_sc.max()))
print("  Test :", float(test_sc.min()),  "→", float(test_sc.max()))

def make_single_step_windows(values: np.ndarray, S: int): # single step window means predict next step
    """
    values: (N,1) scaled series
    returns: X: (N-S, S, 1), y: (N-S, 1)
    """
    N = len(values)
    X = np.stack([values[i:i+S, :] for i in range(N - S)], axis=0)
    y = values[S:, :]  # next value after each window
    return X, y

# Default S for capacity tests; we’ll vary S later in Experiment A
S = 7

X_train_sc, y_train_sc = make_single_step_windows(train_sc, S)
X_val_sc,   y_val_sc   = make_single_step_windows(val_sc,   S)
X_test_sc,  y_test_sc  = make_single_step_windows(test_sc,  S)

BATCH_SIZE = 64

train_ds = TensorDataset(
    torch.tensor(X_train_sc, dtype=torch.float32),
    torch.tensor(y_train_sc, dtype=torch.float32)
)
val_ds = TensorDataset(
    torch.tensor(X_val_sc, dtype=torch.float32),
    torch.tensor(y_val_sc, dtype=torch.float32),
)
test_ds = TensorDataset(
    torch.tensor(X_test_sc, dtype=torch.float32),
    torch.tensor(y_test_sc, dtype=torch.float32),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False)

# Creae a GRU model with single step 
class GRUSingleStep(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, 1)  # linear head for regression in scaled space

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeros(self.gru.num_layers, B, self.gru.hidden_size)
        out, _ = self.gru(x, h0)         # (B,S,H)
        h_last = out[:, -1, :]           # (B,H)
        y_hat  = self.head(h_last)       # (B,1) scaled
        return y_hat

# Define hyper parameters
DEVICE = torch.device("cpu")
EPOCHS = 18
LR     = 1e-3
CLIP   = 1.0
BATCH  = 64

criterion = nn.MSELoss()

# Training loop
def train_one_model(model, train_loader, val_loader, epochs=EPOCHS, Lr = LR, clip=CLIP):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=Lr)
    tr_hist, va_hist = [], []
    for ep in range(1, epochs+1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        # Eval (scaled MSE )
        tr_mse = eval_mse_scaled(model, train_loader)
        va_mse = eval_mse_scaled(model, val_loader)
        tr_hist.append(tr_mse); va_hist.append(va_mse)
        if ep in (1, 6, 12, epochs):
            print(f"[S={S_current}, ep {ep:02d}] train MSE={tr_mse:.5f} | val MSE={va_mse:.5f}")
    return tr_hist, va_hist

@torch.no_grad()
def eval_mse_scaled(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        yhat = model(xb)
        loss = criterion(yhat, yb)
        bs = xb.size(0)
        tot += loss.item() * bs
        n += bs
    return tot/ max(n, 1)

@torch.no_grad()
def predict_scaled(model, X_sc, batch=BATCH):
    model.eval()
    preds = []
    for i in range(0, len(X_sc), batch):
        xb = torch.tensor(X_sc[i:i+batch], dtype=torch.float32).to(DEVICE)
        yhat = model(xb).cpu().numpy()
        preds.append(yhat)
    return np.vstack(preds)  # (N,1) scaled

def inverse_scale_1d(y_scaled):
    flat = y_scaled.reshape(-1, 1)
    back = scaler.inverse_transform(flat)
    return back.ravel()

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))    

results = []
curves  = {}  # S -> (train_curve, val_curve)

S_values = [7, 14, 30]

for S_current in S_values:
    # Rebuild windows (scaled)
    X_train_sc_S, y_train_sc_S = make_single_step_windows(train_sc, S_current)
    X_val_sc_S,   y_val_sc_S   = make_single_step_windows(val_sc,   S_current)
    X_test_sc_S,  y_test_sc_S  = make_single_step_windows(test_sc,  S_current)

    # Build targets in °C to evaluate later
    _, y_val_C_S  = make_single_step_windows(val_np,  S_current)   # (N,1) in °C
    _, y_test_C_S = make_single_step_windows(test_np, S_current)

    # DataLoaders
    train_loader_S = DataLoader(
        TensorDataset(torch.tensor(X_train_sc_S, dtype=torch.float32),
                      torch.tensor(y_train_sc_S, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True, drop_last=False)
    val_loader_S = DataLoader(
        TensorDataset(torch.tensor(X_val_sc_S, dtype=torch.float32),
                      torch.tensor(y_val_sc_S, dtype=torch.float32)),
        batch_size=BATCH, shuffle=False, drop_last=False)
    test_loader_S = DataLoader(
        TensorDataset(torch.tensor(X_test_sc_S, dtype=torch.float32),
                      torch.tensor(y_test_sc_S, dtype=torch.float32)),
        batch_size=BATCH, shuffle=False, drop_last=False)

    # Model
    model = GRUSingleStep(input_size=1, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0)

    # Train briefly
    tr_curve, va_curve = train_one_model(model, train_loader_S, val_loader_S, epochs=EPOCHS, lr=LR, clip=CLIP)
    curves[S_current] = (tr_curve, va_curve)

    # Predict and report in °C (single-step)
    y_val_hat_sc  = predict_scaled(model, X_val_sc_S)[:, 0]   # (N,)
    y_test_hat_sc = predict_scaled(model, X_test_sc_S)[:, 0]
    y_val_hat_C   = inverse_scale_1d(y_val_hat_sc)
    y_test_hat_C  = inverse_scale_1d(y_test_hat_sc)

    # Flatten ground-truth (°C)
    y_val_true_C  = y_val_C_S.ravel()
    y_test_true_C = y_test_C_S.ravel()

    row = {
        "S": S_current,
        "Val_MAE_C":  mae(y_val_true_C,  y_val_hat_C),
        "Val_RMSE_C": rmse(y_val_true_C, y_val_hat_C),
        "Test_MAE_C":  mae(y_test_true_C,  y_test_hat_C),
        "Test_RMSE_C": rmse(y_test_true_C, y_test_hat_C),
    }
    results.append(row)

df_S = pd.DataFrame(results).set_index("S").sort_values("Val_RMSE_C")

best_S = df_S.index[0]
print(f"Best S by Val_RMSE_C: {best_S}")

# Rebuild to get predictions for the chosen S (in case variables were overwritten)
X_val_sc_S, _ = make_single_step_windows(val_sc, best_S)
_, y_val_C_S  = make_single_step_windows(val_np, best_S)

# Refit a fresh model quickly for the slice (or reuse the trained one if kept)
model_bestS = GRUSingleStep(input_size=1, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0)
# A quick re-train for 8 epochs to ensure the object is live (keeps CPU light)
_ , _ = train_one_model(model_bestS,
                        DataLoader(TensorDataset(torch.tensor(make_single_step_windows(train_sc, best_S)[0], dtype=torch.float32),
                                                 torch.tensor(make_single_step_windows(train_sc, best_S)[1], dtype=torch.float32)),
                                   batch_size=BATCH, shuffle=True),
                        DataLoader(TensorDataset(torch.tensor(X_val_sc_S, dtype=torch.float32),
                                                 torch.tensor(make_single_step_windows(val_sc, best_S)[1], dtype=torch.float32)),
                                   batch_size=BATCH, shuffle=False),
                        epochs=8)

y_val_hat_sc = predict_scaled(model_bestS, X_val_sc_S)[:, 0]
y_val_hat_C  = inverse_scale_1d(y_val_hat_sc)
y_val_true_C = y_val_C_S.ravel()

# After getting best S from the first part
# Depth and dropout

# --- Code: Parameterized GRU/LSTM classes (single-step, batch-first) ---
import torch
import torch.nn as nn

class GRUSingleStepCfg(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeros(self.gru.num_layers, B, self.gru.hidden_size)
        out, _ = self.gru(x, h0)     # (B,S,H)
        h_last = out[:, -1, :]       # (B,H)
        return self.head(h_last)     # (B,1), scaled space

class LSTMSingleStepCfg(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
        c0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))  # (B,S,H)
        h_last = out[:, -1, :]           # (B,H)
        return self.head(h_last)         # (B,1), scaled space

# --- Code: Config runner (train, curves, metrics in °C) ---
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Use best S from Experiment A (df_S), else default to 7
try:
    BEST_S = int(df_S.index[0])
except Exception:
    BEST_S = 7
print(f"Using S={BEST_S} for Experiment B")

# Rebuild windows for BEST_S
X_train_sc_S, y_train_sc_S = make_single_step_windows(train_sc, BEST_S)
X_val_sc_S,   y_val_sc_S   = make_single_step_windows(val_sc,   BEST_S)
X_test_sc_S,  y_test_sc_S  = make_single_step_windows(test_sc,  BEST_S)

# Ground-truth in °C for reporting
_, y_val_C_S  = make_single_step_windows(val_np,  BEST_S)
_, y_test_C_S = make_single_step_windows(test_np, BEST_S)
y_val_true_C  = y_val_C_S.ravel()
y_test_true_C = y_test_C_S.ravel()

# DataLoaders
BATCH = 64
train_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_train_sc_S, dtype=torch.float32),
                  torch.tensor(y_train_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=True, drop_last=False)
val_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_val_sc_S, dtype=torch.float32),
                  torch.tensor(y_val_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=False, drop_last=False)
test_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_test_sc_S, dtype=torch.float32),
                  torch.tensor(y_test_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=False, drop_last=False)

DEVICE   = torch.device("cpu")
EPOCHS_B = 18
LR_B     = 1e-3
CLIP_B   = 1.0
criterion = nn.MSELoss()

def train_eval_config(model, label:str, epochs=EPOCHS_B, lr=LR_B, clip=CLIP_B):
    """Train one config; return curves (train/val MSE in *scaled* space) and val/test metrics in °C."""
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    tr_hist, va_hist = [], []

    @torch.no_grad()
    def predict_loader(loader):
        model.eval()
        out = []
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out.append(model(xb).cpu().numpy())
        return np.vstack(out).ravel()

    # train
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader_S:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # scaled MSE for curves
        tr_mse = eval_mse_scaled(model, train_loader_S)
        va_mse = eval_mse_scaled(model, val_loader_S)
        tr_hist.append(tr_mse); va_hist.append(va_mse)

        # quick-on-the-fly real-unit metric for transparency with selection criterion
        if ep in (1, 6, 12, epochs):
            val_pred_sc = predict_loader(val_loader_S)
            val_pred_C  = inverse_scale_1d(val_pred_sc)
            val_rmse_C  = float(np.sqrt(np.mean((y_val_true_C - val_pred_C) ** 2)))

            print(
                f"[{label} | ep {ep:02d}] "
                f"train MSE={tr_mse:.5f} | val MSE (scaled)={va_mse:.5f} "
                f"| val RMSE_C={val_rmse_C:.3f}"
            )

    # predictions (scaled) -> °C
    y_val_hat_sc  = predict_loader(val_loader_S)
    y_test_hat_sc = predict_loader(test_loader_S)
    y_val_hat_C   = inverse_scale_1d(y_val_hat_sc)
    y_test_hat_C  = inverse_scale_1d(y_test_hat_sc)

    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics = {
        "Val_MAE_C":  mae(y_val_true_C,  y_val_hat_C),
        "Val_RMSE_C": rmse(y_val_true_C, y_val_hat_C),
        "Test_MAE_C":  mae(y_test_true_C,  y_test_hat_C),
        "Test_RMSE_C": rmse(y_test_true_C, y_test_hat_C),
    }
    return tr_hist, va_hist, metrics

# --- Code: Config runner (train, curves, metrics in °C) ---
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Use best S from Experiment A (df_S), else default to 7
try:
    BEST_S = int(df_S.index[0])
except Exception:
    BEST_S = 7
print(f"Using S={BEST_S} for Experiment B")

# Rebuild windows for BEST_S
X_train_sc_S, y_train_sc_S = make_single_step_windows(train_sc, BEST_S)
X_val_sc_S,   y_val_sc_S   = make_single_step_windows(val_sc,   BEST_S)
X_test_sc_S,  y_test_sc_S  = make_single_step_windows(test_sc,  BEST_S)

# Ground-truth in °C for reporting
_, y_val_C_S  = make_single_step_windows(val_np,  BEST_S)
_, y_test_C_S = make_single_step_windows(test_np, BEST_S)
y_val_true_C  = y_val_C_S.ravel()
y_test_true_C = y_test_C_S.ravel()

# DataLoaders
BATCH = 64
train_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_train_sc_S, dtype=torch.float32),
                  torch.tensor(y_train_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=True, drop_last=False)
val_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_val_sc_S, dtype=torch.float32),
                  torch.tensor(y_val_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=False, drop_last=False)
test_loader_S = DataLoader(
    TensorDataset(torch.tensor(X_test_sc_S, dtype=torch.float32),
                  torch.tensor(y_test_sc_S, dtype=torch.float32)),
    batch_size=BATCH, shuffle=False, drop_last=False)

DEVICE   = torch.device("cpu")
EPOCHS_B = 18
LR_B     = 1e-3
CLIP_B   = 1.0
criterion = nn.MSELoss()

def train_eval_config(model, label:str, epochs=EPOCHS_B, lr=LR_B, clip=CLIP_B):
    """Train one config; return curves (train/val scaled MSE) and val/test metrics in °C."""
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    tr_hist, va_hist = [], []

    # train
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader_S:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # scaled MSE for curves
        tr_mse = eval_mse_scaled(model, train_loader_S)
        va_mse = eval_mse_scaled(model, val_loader_S)
        tr_hist.append(tr_mse); va_hist.append(va_mse)
        if ep in (1, 6, 12, epochs):
            print(f"[{label} | ep {ep:02d}] train MSE={tr_mse:.5f} | val MSE={va_mse:.5f}")

    # predictions (scaled) -> °C
    @torch.no_grad()
    def predict_loader(loader):
        model.eval()
        out = []
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out.append(model(xb).cpu().numpy())
        return np.vstack(out).ravel()

    y_val_hat_sc  = predict_loader(val_loader_S)
    y_test_hat_sc = predict_loader(test_loader_S)
    y_val_hat_C   = inverse_scale_1d(y_val_hat_sc)
    y_test_hat_C  = inverse_scale_1d(y_test_hat_sc)

    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics = {
        "Val_MAE_C":  mae(y_val_true_C,  y_val_hat_C),
        "Val_RMSE_C": rmse(y_val_true_C, y_val_hat_C),
        "Test_MAE_C":  mae(y_test_true_C,  y_test_hat_C),
        "Test_RMSE_C": rmse(y_test_true_C, y_test_hat_C),
    }
    return tr_hist, va_hist, metrics

# --- Code: Run the grid and collect results ---
import copy

# Define the small grid
grid = [(cell, layers, dropout)
        for cell in ["GRU", "LSTM"]
        for layers in [1, 2]
        for dropout in [0.0, 0.2]]  # dropout ignored by PyTorch when layers == 1

curves_B = {}   # label -> (train_curve, val_curve)
rows_B   = []

# Best-model trackers
best_val_rmseC = float("inf")
best_label = None
best_state = None
best_val_mse_scaled = None
model_best = None

for (cell, layers, dropout) in grid:
    label = f"{cell}-L{layers}-do{dropout}"

    if cell == "GRU":
        model = GRUSingleStepCfg(
            input_size=1, hidden_size=64, num_layers=layers,
            dropout=dropout, batch_first=True
        )
    else:
        model = LSTMSingleStepCfg(
            input_size=1, hidden_size=64, num_layers=layers,
            dropout=dropout, batch_first=True
        )

    # Train & evaluate
    tr_curve, va_curve, m = train_eval_config(
        model, label=label, epochs=EPOCHS_B, lr=LR_B, clip=CLIP_B
    )

    # Save curves and row for summary table
    curves_B[label] = (tr_curve, va_curve)
    rows_B.append({"Cell": cell, "Layers": layers, "Dropout": dropout, **m})

    # Running best (compare on Val_RMSE_C in °C)
    if m["Val_RMSE_C"] < best_val_rmseC:
        best_val_rmseC = m["Val_RMSE_C"]
        best_label = label
        best_val_mse_scaled = float(va_curve[-1])  # last-epoch scaled val MSE
        model_best = copy.deepcopy(model)
        best_state = copy.deepcopy(model.state_dict())
        print(
            f"New best model: {best_label} "
            f"with Val_RMSE_C={best_val_rmseC:.3f} "
            f"(val MSE scaled={best_val_mse_scaled:.5f})"
        )

# Ensure model_best has the exact best weights
if best_state is not None:
    model_best.load_state_dict(best_state)

# Summary table
df_B = pd.DataFrame(rows_B)
df_B_sorted = df_B.sort_values(
    ["Val_RMSE_C", "Val_MAE_C"], ascending=[True, True]
).reset_index(drop=True)

# --- Code: Run the grid and collect results ---
grid = []
for cell in ["GRU", "LSTM"]:
    for layers in [1, 2]:
        for dropout in [0.0, 0.2]:
            # PyTorch ignores dropout if layers == 1; we still list it for completeness
            grid.append((cell, layers, dropout))

curves_B = {}   # label -> (train_curve, val_curve)
rows_B   = []
# save best model
model_best = None

for (cell, layers, dropout) in grid:
    label = f"{cell}-L{layers}-do{dropout}"
    if cell == "GRU":
        model = GRUSingleStepCfg(input_size=1, hidden_size=64, num_layers=layers, dropout=dropout, batch_first=True)
    else:
        model = LSTMSingleStepCfg(input_size=1, hidden_size=64, num_layers=layers, dropout=dropout, batch_first=True)

    tr_curve, va_curve, m = train_eval_config(model, label=label, epochs=EPOCHS_B, lr=LR_B, clip=CLIP_B)
    # save best model
    if model_best is None or m["Val_RMSE_C"] < rows_B[0]["Val_RMSE_C"]:
        model_best = model
        print(f"New best model: {label} with Val_RMSE_C={m['Val_RMSE_C']:.3f}")
    curves_B[label] = (tr_curve, va_curve)
    row = {"Cell": cell, "Layers": layers, "Dropout": dropout}
    row.update(m)
    rows_B.append(row)

df_B = pd.DataFrame(rows_B)
df_B_sorted = df_B.sort_values(["Val_RMSE_C", "Val_MAE_C"], ascending=[True, True]).reset_index(drop=True)

