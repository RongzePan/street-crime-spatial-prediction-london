"""
LSTM spatio‑temporal crime prediction model (PyTorch).
Input: feature vectors for each LSOA over the past SEQ_LEN months.
Output: crime density prediction for the next month.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from config import (PROC_DIR, MODEL_DIR, RESULT_DIR, RANDOM_SEED,
                    SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
                    LR, EPOCHS, BATCH_SIZE, PATIENCE,
                    TRAIN_END, VAL_END, HOTSPOT_PCT)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Fix 2: removed crime_rate_1k (all NaN), use non‑NaN features ─────────
FEATURES = [
    "crime_count",            # absolute crime count (non‑NaN)
    "crime_density",          # crimes per km² (non‑NaN, computed from area_m2)
    "month_of_year_sin",
    "month_of_year_cos",
    "year_norm",
]
TARGET = "crime_density"      # Fix 2: crime_rate_1k → crime_density


class CrimeDataset(Dataset):
    """
    Each sample = (lsoa, starting month),
    X = [seq_len, n_features], y = next month's crime density (scalar).
    """
    def __init__(self, sequences, targets):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN_SIZE,
                 layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def build_sequences(full_panel: pd.DataFrame,
                    scaler: StandardScaler,
                    fit_scaler: bool = False,

                    features: list = None,
                    target: str = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Fix 1: Build all sequences from the full panel, then let the caller split
    by target month.

    Original issue:
      train_df = panel[panel["month"] <= TRAIN_END]  ← only 20 months
      X_tr, y_tr = build_sequences(train_df, ...)    ← val_df has only 6 months < SEQ_LEN=12 → 0 sequences

    New approach:
      1. Build sequences from the full 36‑month panel.
      2. Use only training‑period data to fit the scaler (no data leakage).
      3. Transform the entire panel using training‑period statistics.
      4. Return all sequences together with their target months; run_lstm will
         mask them by month.
         → val sequences can correctly use training‑period data as the lookback window.

    Build LSTM sequences from panel data (parameterised features and target).

    Parameters
    ----------
    full_panel : pd.DataFrame
        Original panel data
    scaler : StandardScaler
        Scaler for feature standardisation
    fit_scaler : bool
        Whether to fit the scaler on training‑period data
    features : list, optional
        List of feature column names (defaults to module‑level FEATURES)
    target : str, optional
        Target column name (defaults to module‑level TARGET)

    """
    if features is None:
        features = FEATURES
    if target is None:
        target = TARGET

    full_panel = full_panel.copy()
    full_panel["month"] = pd.to_datetime(full_panel["month"])

    # ── Fit scaler only on training‑period data (prevent data leakage) ──

    if fit_scaler:
        train_data = full_panel[
            full_panel["month"] <= pd.to_datetime(TRAIN_END)
        ]
        if len(train_data) == 0:
            raise ValueError(
                f"Training‑period data is empty! TRAIN_END='{TRAIN_END}', "
                f"actual data range: "
                f"{full_panel['month'].min().strftime('%Y-%m')} ~ "
                f"{full_panel['month'].max().strftime('%Y-%m')}\n"
                "Please update TRAIN_END in config.py."
            )
        scaler.fit(train_data[features].fillna(0).values)
        print(f"  Scaler fitted on training period "
              f"{train_data['month'].min().strftime('%Y-%m')} ~ "
              f"{train_data['month'].max().strftime('%Y-%m')}")

    # ── Transform the full panel using training‑period statistics ─────────
    panel_scaled = full_panel.copy()
    panel_scaled[features] = scaler.transform(
        full_panel[features].fillna(0).values
    )

    # ── Build all sequences ──────────────────────────────────────────────
    seqs, tgts, meta = [], [], []
    for lsoa_code, grp in panel_scaled.groupby("lsoa_code"):
        grp    = grp.sort_values("month").reset_index(drop=True)
        X      = grp[features].values   # (T, n_features)
        y      = grp[target].values     # (T,)
        months = grp["month"].values    # (T,)

        for i in range(len(X) - SEQ_LEN):
            seqs.append(X[i: i + SEQ_LEN])
            tgts.append(y[i + SEQ_LEN])
            meta.append({
                "lsoa_code": lsoa_code,
                "month":     months[i + SEQ_LEN],
            })

    if not seqs:
        raise ValueError(
            f"Sequence construction empty! SEQ_LEN={SEQ_LEN}, "
            f"number of months in data = {full_panel['month'].nunique()}.\n"
            f"Each LSOA needs at least {SEQ_LEN + 1} months of data."
        )

    return (
        np.array(seqs,  dtype=np.float32),
        np.array(tgts,  dtype=np.float32),
        pd.DataFrame(meta),
    )


def train_model(model, train_loader, val_loader, device):
    """
    Fix 3: When val_loader is empty, skip validation and monitor training loss
    for early stopping, avoiding division‑by‑zero with `val_loss /= len(val_loader.dataset)`.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    best_loss, best_state, no_improve = float("inf"), None, 0
    has_val = len(val_loader.dataset) > 0

    if not has_val:
        print("  ⚠ Validation set empty; using training loss as early‑stopping monitor")

    for epoch in range(EPOCHS):

        # ── Diagnostic checkpoint 3: confirm model is on the correct device
        if epoch == 0:
            print(f"Model device before training: {next(model.parameters()).device}")

        # ── Train ──────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:

            # ── Diagnostic checkpoint 4: confirm data moved to device ──
            if not hasattr(train_loader, '_first_batch_checked'):
                print(f"Training batch X device: {X.device} (after moving to {device})")
                train_loader._first_batch_checked = True

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train_loader.dataset)

        # ── Validate (skip if empty) ──────────────────────────
        if has_val:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    va_loss += criterion(model(X), y).item() * len(y)
            va_loss /= len(val_loader.dataset)   # safe because len > 0
            monitor = va_loss
            val_str = f"{va_loss:.4f}"
        else:
            va_loss = float("nan")
            monitor = tr_loss                    # monitor training loss when no validation set
            val_str = "N/A"

        scheduler.step(monitor)

        if monitor < best_loss:
            best_loss  = monitor
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"train={tr_loss:.4f} | val={val_str}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(best_state or model.state_dict(),
               MODEL_DIR / "lstm_best.pt")
    return model

def run_lstm(panel: pd.DataFrame,
             target_col: str = "crime_density",
             count_col: str = "crime_count") -> pd.DataFrame:
    """
    Run LSTM spatio‑temporal prediction for an arbitrary crime type.

    Parameters
    ----------
    panel : pd.DataFrame
        Pre‑processed panel data
    target_col : str
        Target density column name, e.g. 'density_theft_from_person' or 'crime_density'
    count_col : str
        Corresponding count column name, e.g. 'n_theft_from_person' or 'crime_count'
    """
    # ── Diagnostic checkpoint 1: PyTorch and CUDA information ──────────
    print("=" * 50)
    print("GPU Availability Diagnostics")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (PyTorch compile time): {torch.version.cuda}")
    else:
        print("CUDA not available! Possible reasons:")
        print("  1. GPU version of PyTorch not installed (use `pip install torch ... --index-url ...` to install CUDA version)")
        print("  2. NVIDIA driver not installed or too old")
        print("  3. PyTorch and CUDA versions incompatible")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"])

    # ── Dynamically build feature list ──────────────────────────────────────────
    # Base time features (always included)
    time_features = ["month_of_year_sin", "month_of_year_cos", "year_norm"]
    # Add count_col and target_col to the feature set (if different)
    features = [count_col, target_col] + time_features
    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))
    target = target_col

    # Diagnostic: check that columns exist
    missing = [c for c in features if c not in panel.columns]
    if missing:
        raise ValueError(f"The following feature columns are missing from the panel: {missing}")
    if target not in panel.columns:
        raise ValueError(f"Target column '{target}' does not exist in the panel.")

    # Diagnostic 2: check for high NaN rates (optional warning, not fatal)
    for col in features + [target]:
        if col in panel.columns:
            nr = panel[col].isna().mean()
            if nr > 0.05:
                print(f"  ⚠ '{col}' has high NaN rate: {nr:.1%}")

    print(f"  Feature columns: {features}")
    print(f"  Target column: {target}")

    # ── Fix 1: Build all sequences from the full panel, then split by month ──
    scaler = StandardScaler()
    X_all, y_all, meta_all = build_sequences(
        panel, scaler, fit_scaler=True,
        features=features, target=target
    )

    train_end_ts = pd.to_datetime(TRAIN_END)
    val_end_ts   = pd.to_datetime(VAL_END)
    seq_months   = pd.to_datetime(meta_all["month"])

    tr_mask = seq_months <= train_end_ts
    va_mask = (seq_months > train_end_ts) & (seq_months <= val_end_ts)
    te_mask = seq_months > val_end_ts

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]
    X_te, y_te = X_all[te_mask], y_all[te_mask]
    meta_te    = meta_all[te_mask].reset_index(drop=True)

    print(f"Train: {len(y_tr):,} | Val: {len(y_va):,} | Test: {len(y_te):,}")

    if len(y_tr) == 0:
        raise ValueError(
            f"Training sequences empty. TRAIN_END='{TRAIN_END}', SEQ_LEN={SEQ_LEN}, "
            f"need at least {SEQ_LEN+1} training months."
        )

    tr_loader = DataLoader(CrimeDataset(X_tr, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(CrimeDataset(X_va, y_va),
                           batch_size=BATCH_SIZE)
    te_loader = DataLoader(CrimeDataset(X_te, y_te),
                           batch_size=BATCH_SIZE)

    model  = LSTMPredictor(n_features=len(features)).to(device)

    # ── Diagnostic checkpoint 2: confirm device of model parameters ──
    print(f"Model first parameter device: {next(model.parameters()).device}")

    model  = train_model(model, tr_loader, va_loader, device)

    # ── Test set predictions ─────────────────────────────────────────────────
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in te_loader:
            preds.extend(model(X.to(device)).cpu().numpy().tolist())

    meta_te["predicted_density"] = preds
    meta_te["actual_density"]    = y_te

    # ── PAI computation (using crime_count, consistent with KDE) ────────────
    # Obtain actual crime counts for the test period from the original panel
    panel_test = panel[
        panel["month"] > val_end_ts
    ][["lsoa_code", "month", count_col]].copy()
    panel_test.rename(columns={count_col: "crime_count"}, inplace=True)

    # LSOA areas (for area_pct, same logic as KDE)
    lsoa_area  = (panel[["lsoa_code", "area_m2"]]
                  .drop_duplicates("lsoa_code")
                  .set_index("lsoa_code")["area_m2"])
    total_area = lsoa_area.sum()

    results = []
    for month, grp in meta_te.groupby("month"):
        ts        = pd.Timestamp(month)
        month_str = ts.strftime("%Y-%m")

        # Rank by predicted density → hotspot
        threshold = grp["predicted_density"].quantile(1 - HOTSPOT_PCT)
        grp = grp.copy()
        grp["predicted_hotspot"] = grp["predicted_density"] >= threshold
        hotspot_set  = set(grp.loc[grp["predicted_hotspot"], "lsoa_code"])
        hotspot_area = lsoa_area.reindex(hotspot_set).sum()
        area_pct     = hotspot_area / total_area if total_area > 0 else 0

        # Actual crimes (crime_count)
        actual = panel_test[panel_test["month"] == ts]
        actual_total = int(actual["crime_count"].sum())
        if actual_total == 0:
            continue

        captured     = int(actual[actual["lsoa_code"].isin(hotspot_set)]
                           ["crime_count"].sum())
        capture_rate = captured / actual_total
        pai          = capture_rate / area_pct if area_pct > 0 else 0

        results.append({
            "month":           ts,
            "model":           "LSTM",
            "pai":             pai,
            "capture_rate":    capture_rate,
            "area_pct":        area_pct,
            "crimes_captured": captured,
            "crimes_total":    actual_total,
        })
        print(f"  LSTM {month_str}: PAI={pai:.3f}  "
              f"capture={capture_rate:.1%}  area={area_pct:.1%}")

    df = pd.DataFrame(results)
    if not df.empty:

        out_name = f"lstm_results_{target_col.replace('density_', '') if target_col.startswith('density_') else 'all'}.parquet"

        df.to_parquet(RESULT_DIR / "lstm_results.parquet", index=False)
        print(f"\n[LSTM] Complete — mean PAI = {df['pai'].mean():.3f}  "
              f"(min={df['pai'].min():.3f}, max={df['pai'].max():.3f})")
    else:
        print("[LSTM] No valid results; please check test month range.")

    return df