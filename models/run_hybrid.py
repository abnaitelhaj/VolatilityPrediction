"""HAR + LSTM residual hybrid under Protocol B.

Stage 1: fit HAR by OLS (frozen).
Stage 2: LSTM predicts the HAR residual e = y - yhat_HAR.
Final forecast: yhat = yhat_HAR + w * ehat_LSTM.
Head is zero-initialized so training starts as pure HAR.
w is chosen on validation; test is scored once after train+val refit.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.dates import DateFormatter
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT / "eval_freeze"))

from protocol import (  # noqa: E402
    TARGET_COL,
    add_split_column,
    load_model_ready,
    metrics_from_log_forecast,
)

SEED = 42
DEVICE = torch.device("cpu")
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 64
GRID_L = [22, 60]
GRID_HIDDEN = [64, 128]
GRID_DEPTH = [1, 2]
GRID_DROPOUT = [0.1]
GRID_LR = [1e-3, 3e-4]
GRID_W = [0.2, 0.4, 0.6, 0.8, 1.0]
CHANNELS = ["log_rv", "log_rv_w", "log_rv_m", "ret_d_sq", "volume_z30"]
HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]

C_ACTUAL = "#111827"
C_HAR = "#2563EB"
C_LSTM = "#7C3AED"
C_HYB = "#059669"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


@dataclass(frozen=True)
class HParams:
    L: int
    hidden: int
    depth: int
    dropout: float
    lr: float


class ResidualLSTM(nn.Module):
    def __init__(self, hp: HParams, n_in: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=hp.hidden,
            num_layers=hp.depth,
            batch_first=True,
            dropout=hp.dropout if hp.depth > 1 else 0.0,
        )
        self.drop = nn.Dropout(hp.dropout)
        self.head = nn.Linear(hp.hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def make_windows(values: np.ndarray, targets: np.ndarray, L: int, idx: np.ndarray):
    if values.ndim == 1:
        values = values[:, None]
    xs, ys, keep = [], [], []
    for i in idx:
        if i - L + 1 < 0 or not np.isfinite(targets[i]):
            continue
        block = values[i - L + 1 : i + 1]
        if not np.isfinite(block).all():
            continue
        xs.append(block)
        ys.append(targets[i])
        keep.append(int(i))
    c = values.shape[1]
    if not xs:
        return (
            np.empty((0, L, c), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(keep, dtype=np.int64),
    )


def scale_channels(raw: np.ndarray, fit_idx: np.ndarray):
    mu = raw[fit_idx].mean(axis=0)
    sd = raw[fit_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32), mu, sd


def loader(X: np.ndarray, y: np.ndarray, *, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def predict(model: ResidualLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.empty((0,), dtype=np.float64)
    out = []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_one(
    hp: HParams,
    n_in: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray | None,
    y_va: np.ndarray | None,
    *,
    max_epochs: int,
    patience: int | None,
) -> tuple[ResidualLSTM, int, float]:
    set_seed(SEED)
    model = ResidualLSTM(hp, n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = np.inf
    best_epoch = 0
    bad = 0
    tr_loader = loader(X_tr, y_tr, shuffle=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if X_va is not None and y_va is not None and len(X_va):
            val_rmse = float(np.sqrt(np.mean((predict(model, X_va) - y_va) ** 2)))
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if patience is not None and bad >= patience:
                break
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = float(np.sqrt(np.mean((predict(model, X_tr) - y_tr) ** 2)))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, float(best_val)


def fit_har(df: pd.DataFrame, fit_idx: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(fit_idx)), df.iloc[fit_idx][HAR_COLS].to_numpy(dtype=float)])
    y = df.iloc[fit_idx][TARGET_COL].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    X_all = np.column_stack([np.ones(len(df)), df[HAR_COLS].to_numpy(dtype=float)])
    return X_all @ beta


def prepare(raw, resid, L, fit_idx, train_idx, val_idx, test_idx):
    scaled, mu, sd = scale_channels(raw, fit_idx)
    X_tr, y_tr, i_tr = make_windows(scaled, resid, L, train_idx)
    out = {"mu": mu, "sd": sd, "X_tr": X_tr, "y_tr": y_tr, "i_tr": i_tr}
    if val_idx is not None:
        X_va, y_va, i_va = make_windows(scaled, resid, L, val_idx)
        out.update({"X_va": X_va, "y_va": y_va, "i_va": i_va})
    if test_idx is not None:
        X_te, y_te, i_te = make_windows(scaled, resid, L, test_idx)
        out.update({"X_te": X_te, "y_te": y_te, "i_te": i_te})
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    df = add_split_column(load_model_ready())
    y = df[TARGET_COL].to_numpy(dtype=float)
    split = df["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    fit_idx = np.where(np.isin(split, ["train", "val"]))[0]
    raw = df[CHANNELS].to_numpy(dtype=float)
    n_in = raw.shape[1]

    # Tune-stage HAR on train only
    har_tune = fit_har(df, train_idx)
    resid_tune = y - har_tune

    grid = [
        HParams(L=L, hidden=h, depth=d, dropout=drop, lr=lr)
        for L, h, d, drop, lr in product(GRID_L, GRID_HIDDEN, GRID_DEPTH, GRID_DROPOUT, GRID_LR)
    ]
    print(f"hybrid grid size={len(grid)}")
    results = []
    best = None

    for k, hp in enumerate(grid, 1):
        data = prepare(raw, resid_tune, hp.L, train_idx, train_idx, val_idx, None)
        model, best_epoch, best_val = train_one(
            hp, n_in, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"],
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        # pick w on val using blended forecast RMSE
        ehat_va = predict(model, data["X_va"])
        y_va = y[data["i_va"]]
        har_va = har_tune[data["i_va"]]
        best_w, best_blend = GRID_W[0], np.inf
        for w in GRID_W:
            yhat = har_va + w * ehat_va
            rmse = float(np.sqrt(np.mean((yhat - y_va) ** 2)))
            if rmse < best_blend:
                best_blend, best_w = rmse, w
        row = {
            **asdict(hp),
            "best_epoch": best_epoch,
            "val_resid_rmse": best_val,
            "val_blend_rmse": best_blend,
            "w": best_w,
        }
        results.append(row)
        print(
            f"[{k:02d}/{len(grid)}] L={hp.L} h={hp.hidden} d={hp.depth} "
            f"lr={hp.lr:.0e} w={best_w:.1f} → val_blend_rmse={best_blend:.4f} epoch={best_epoch}"
        )
        if best is None or best_blend < best["val_blend_rmse"]:
            best = {**row}

    assert best is not None
    hp = HParams(best["L"], best["hidden"], best["depth"], best["dropout"], best["lr"])
    w = float(best["w"])
    epochs = int(best["best_epoch"])

    # Final: HAR on train+val, residual LSTM refit for chosen epochs
    har_final = fit_har(df, fit_idx)
    resid_final = y - har_final
    data_f = prepare(raw, resid_final, hp.L, fit_idx, fit_idx, None, test_idx)
    model, _, _ = train_one(
        hp, n_in, data_f["X_tr"], data_f["y_tr"], None, None,
        max_epochs=epochs, patience=None,
    )
    ehat_te = predict(model, data_f["X_te"])
    i_te = data_f["i_te"]
    yhat_hyb = har_final[i_te] + w * ehat_te
    y_te = y[i_te]

    board = {
        "HAR": metrics_from_log_forecast(y_te, har_final[i_te]),
        "Hybrid": metrics_from_log_forecast(y_te, yhat_hyb),
    }
    # standalone LSTM test from saved forecasts for the table
    lstm_fc = pd.read_parquet(OUT / "lstm_forecasts.parquet")
    lstm_te = lstm_fc[lstm_fc["split"] == "test"].dropna(subset=["yhat_lstm"])
    board["LSTM"] = metrics_from_log_forecast(
        lstm_te["y"].to_numpy(), lstm_te["yhat_lstm"].to_numpy()
    )

    meta = {
        "selected": {**best, "w": w, "final_epochs": epochs},
        "channels": CHANNELS,
        "test": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in board.items()},
        "grid": results,
    }
    (OUT / "hybrid_stats.json").write_text(json.dumps(meta, indent=2))

    out_df = df[["date_utc", "split", TARGET_COL]].copy()
    out_df["yhat_har"] = har_final
    out_df["yhat_hybrid"] = np.nan
    out_df.loc[out_df.index[i_te], "yhat_hybrid"] = yhat_hyb
    # also fill train/val hybrid for completeness after final refit
    data_all = prepare(raw, resid_final, hp.L, fit_idx, fit_idx, None, None)
    # predict all windows from final model
    scaled, _, _ = scale_channels(raw, fit_idx)
    all_idx = np.arange(hp.L - 1, len(y))
    X_all, _, i_all = make_windows(scaled, resid_final, hp.L, all_idx)
    ehat_all = predict(model, X_all)
    out_df.loc[out_df.index[i_all], "yhat_hybrid"] = har_final[i_all] + w * ehat_all
    out_df.to_parquet(OUT / "hybrid_forecasts.parquet", index=False)

    # metrics bars
    labels = ["HAR", "LSTM", "Hybrid"]
    colors = [C_HAR, C_LSTM, C_HYB]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.6))
    for ax, key, title in zip(
        axes,
        ["rmse_log", "mae_log", "mape_rv", "qlike"],
        ["RMSE (log)", "MAE (log)", "MAPE (RV, %)", "QLIKE"],
    ):
        vals = [board[m][key] for m in labels]
        ax.bar(labels, vals, color=colors, width=0.65, edgecolor="white", linewidth=0.7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Test metrics: HAR, LSTM, Hybrid", y=1.02, fontsize=13, fontweight="bold")
    save(fig, "hybrid_test_metrics.png")

    # paths
    tdf = df.iloc[i_te].copy()
    tdf["yhat_hybrid"] = yhat_hyb
    tdf["yhat_har"] = har_final[i_te]
    tdf = tdf.merge(lstm_te[["date_utc", "yhat_lstm"]], on="date_utc", how="left")

    def plot_path(mask, title, outfile):
        d = tdf.loc[mask]
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        ax.plot(d["date_utc"], d[TARGET_COL], color=C_ACTUAL, lw=1.3, label="Actual")
        ax.plot(d["date_utc"], d["yhat_har"], color=C_HAR, lw=1.1, label="HAR")
        ax.plot(d["date_utc"], d["yhat_lstm"], color=C_LSTM, lw=1.1, label="LSTM")
        ax.plot(d["date_utc"], d["yhat_hybrid"], color=C_HYB, lw=1.1, label="Hybrid")
        ax.set_ylabel(r"$\log\mathrm{RV}$")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=4, fontsize=9)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        save(fig, outfile)

    dates = pd.to_datetime(tdf["date_utc"], utc=True)
    plot_path(
        (dates >= dates.iloc[0]) & (dates < dates.iloc[0] + pd.DateOffset(months=2)),
        "HAR / LSTM / Hybrid (first two test months)",
        "hybrid_test_paths_2mo.png",
    )
    plot_path(dates >= pd.Timestamp("2026-01-01", tz="UTC"), "HAR / LSTM / Hybrid (2026)", "hybrid_test_paths_2026.png")

    print(json.dumps(meta["selected"], indent=2))
    print(json.dumps(meta["test"], indent=2))


if __name__ == "__main__":
    main()
