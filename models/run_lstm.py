"""LSTM under eval_freeze Protocol B.

Forecast is HAR plus an LSTM residual:
  yhat = HAR(log RV lags) + w * LSTM(window of log RV, R^2, volume z).
Tune L / hidden / depth / dropout / LR / blend w on val, then refit
train+val with frozen settings. Score test once.
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
    LOG_RV_COL,
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

# Small grid — chronological val pick, not k-fold CV
GRID_L = [22, 60]
GRID_HIDDEN = [64]
GRID_DEPTH = [1, 2]
GRID_DROPOUT = [0.1, 0.2]
GRID_LR = [1e-3, 3e-4]
HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]
CHANNELS = ["log_rv", "log_rv_w", "log_rv_m", "ret_d_sq", "volume_z30"]
BLEND_W = np.linspace(0.0, 2.0, 41)

C_ACTUAL = "#111827"
C_NAIVE = "#6B7280"
C_LSTM = "#7C3AED"
C_HAR = "#2563EB"

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


class VolLSTM(nn.Module):
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
        # Start at a zero residual so the forecast begins as HAR.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        return self.head(last).squeeze(-1)


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


def make_windows(
    values: np.ndarray,
    targets: np.ndarray,
    L: int,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (N, L, C) windows ending at each index in idx (inclusive)."""
    if values.ndim == 1:
        values = values[:, None]
    xs, ys, keep = [], [], []
    for i in idx:
        if i - L + 1 < 0:
            continue
        xs.append(values[i - L + 1 : i + 1])
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


def fit_har(df: pd.DataFrame, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(fit_idx)), df.iloc[fit_idx][HAR_COLS].to_numpy(dtype=float)])
    y = df.iloc[fit_idx][TARGET_COL].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    X_all = np.column_stack([np.ones(len(df)), df[HAR_COLS].to_numpy(dtype=float)])
    return X_all @ beta, beta


def scale_channels(raw: np.ndarray, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = raw[fit_idx].mean(axis=0)
    sd = raw[fit_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32), mu, sd


def best_blend(har: np.ndarray, resid_hat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pick residual weight on val. w=0 is pure HAR."""
    best_w, best_rmse = 0.0, float(np.sqrt(np.mean((har - y) ** 2)))
    for w in BLEND_W:
        yhat = har + float(w) * resid_hat
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        if rmse < best_rmse - 1e-8:
            best_w, best_rmse = float(w), rmse
    return best_w, best_rmse


def scale_fit(train_vals: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(train_vals))
    sd = float(np.std(train_vals))
    if sd < 1e-8:
        sd = 1.0
    return mu, sd


def apply_scale(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / sd


def loader(X: np.ndarray, y: np.ndarray, *, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def predict(model: VolLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.empty((0,), dtype=np.float64)
    out = []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def eval_rmse(model: VolLSTM, X: np.ndarray, y: np.ndarray) -> float:
    yhat = predict(model, X)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


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
    har_va: np.ndarray | None = None,
    y_va_level: np.ndarray | None = None,
) -> tuple[VolLSTM, int, float, float, list[dict]]:
    """Train residual LSTM. Early-stop on blended val forecast RMSE."""
    set_seed(SEED)
    model = VolLSTM(hp, n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    history: list[dict] = []
    best_state = None
    best_val = np.inf
    best_epoch = 0
    best_w = 1.0
    bad = 0

    tr_loader = loader(X_tr, y_tr, shuffle=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        train_rmse = float(np.sqrt(total / max(n, 1)))
        row: dict = {"epoch": epoch, "train_rmse": train_rmse}
        if X_va is not None and y_va is not None and len(X_va):
            resid_hat = predict(model, X_va)
            if har_va is not None and y_va_level is not None:
                w, val_rmse = best_blend(har_va, resid_hat, y_va_level)
            else:
                w, val_rmse = 1.0, float(np.sqrt(np.mean((resid_hat - y_va) ** 2)))
            row["val_rmse"] = val_rmse
            row["w"] = w
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_epoch = epoch
                best_w = w
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            history.append(row)
            if patience is not None and bad >= patience:
                break
        else:
            history.append(row)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = train_rmse

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, float(best_val), float(best_w), history


def prepare_scaled_series(
    raw: np.ndarray,
    resid: np.ndarray,
    level: np.ndarray,
    har: np.ndarray,
    L: int,
    fit_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
) -> dict:
    scaled, mu, sd = scale_channels(raw, fit_idx)
    X_tr, y_tr, i_tr = make_windows(scaled, resid, L, train_idx)
    out = {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "X_tr": X_tr,
        "y_tr": y_tr,
        "i_tr": i_tr,
        "har_tr": har[i_tr],
        "level_tr": level[i_tr],
    }
    if val_idx is not None:
        X_va, y_va, i_va = make_windows(scaled, resid, L, val_idx)
        out.update({
            "X_va": X_va,
            "y_va": y_va,
            "i_va": i_va,
            "har_va": har[i_va],
            "level_va": level[i_va],
        })
    if test_idx is not None:
        X_te, y_te, i_te = make_windows(scaled, resid, L, test_idx)
        out.update({
            "X_te": X_te,
            "y_te": y_te,
            "i_te": i_te,
            "har_te": har[i_te],
            "level_te": level[i_te],
        })
    all_idx = np.arange(L - 1, len(resid))
    X_all, y_all, i_all = make_windows(scaled, resid, L, all_idx)
    out.update({"X_all": X_all, "y_all": y_all, "i_all": i_all, "har_all_idx": har[i_all]})
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    df = add_split_column(load_model_ready())
    level = df[TARGET_COL].to_numpy(dtype=float)
    split = df["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    raw = df[CHANNELS].to_numpy(dtype=float)
    n_in = raw.shape[1]

    har_tune, beta_tune = fit_har(df, train_idx)
    resid_tune = level - har_tune

    grid = [
        HParams(L=L, hidden=h, depth=d, dropout=drop, lr=lr)
        for L, h, d, drop, lr in product(
            GRID_L, GRID_HIDDEN, GRID_DEPTH, GRID_DROPOUT, GRID_LR
        )
    ]

    print(f"grid size={len(grid)} device={DEVICE} channels={CHANNELS}")
    results: list[dict] = []
    best: dict | None = None

    for k, hp in enumerate(grid, 1):
        data = prepare_scaled_series(
            raw, resid_tune, level, har_tune, hp.L, train_idx, train_idx, val_idx, None
        )
        model, best_epoch, best_val, best_w, hist = train_one(
            hp,
            n_in,
            data["X_tr"],
            data["y_tr"],
            data["X_va"],
            data["y_va"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            har_va=data["har_va"],
            y_va_level=data["level_va"],
        )
        row = {
            **asdict(hp),
            "best_epoch": best_epoch,
            "blend_w": best_w,
            "val_rmse": best_val,
            "n_train": int(len(data["y_tr"])),
            "n_val": int(len(data["y_va"])),
            "epochs_ran": len(hist),
        }
        results.append(row)
        print(
            f"[{k:02d}/{len(grid)}] L={hp.L} h={hp.hidden} d={hp.depth} "
            f"drop={hp.dropout} lr={hp.lr:.0e} w={best_w:.2f} → val_rmse={best_val:.4f} "
            f"epoch={best_epoch}"
        )
        if best is None or best_val < best["val_rmse"]:
            best = {**row}

    assert best is not None
    best_hp = HParams(
        L=int(best["L"]),
        hidden=int(best["hidden"]),
        depth=int(best["depth"]),
        dropout=float(best["dropout"]),
        lr=float(best["lr"]),
    )
    epoch_budget = max(int(best["best_epoch"]), 1)
    blend_w = float(best["blend_w"])

    # Final refit on train+val for epoch_budget epochs (no early stop)
    fit_idx = np.where(np.isin(split, ["train", "val"]))[0]
    har_final, beta_final = fit_har(df, fit_idx)
    resid_final = level - har_final
    final_data = prepare_scaled_series(
        raw, resid_final, level, har_final, best_hp.L, fit_idx, fit_idx, None, test_idx
    )
    final_model, _, _, _, _ = train_one(
        best_hp,
        n_in,
        final_data["X_tr"],
        final_data["y_tr"],
        None,
        None,
        max_epochs=epoch_budget,
        patience=None,
    )

    resid_te = predict(final_model, final_data["X_te"])
    yhat_test = final_data["har_te"] + blend_w * resid_te
    y_test = final_data["level_te"]
    test_metrics = metrics_from_log_forecast(y_test, yhat_test)

    resid_all = predict(final_model, final_data["X_all"])
    yhat_series = pd.Series(np.nan, index=df.index, dtype=float)
    yhat_series.iloc[final_data["i_all"]] = final_data["har_all_idx"] + blend_w * resid_all

    naive_yhat = df.loc[final_data["i_te"], LOG_RV_COL].to_numpy(dtype=float)
    naive_metrics = metrics_from_log_forecast(y_test, naive_yhat)
    har_yhat = final_data["har_te"]
    har_metrics = metrics_from_log_forecast(y_test, har_yhat)
    har_all = har_final

    stats = {
        "seed": SEED,
        "device": str(DEVICE),
        "channels": CHANNELS,
        "model": "HAR + w * LSTM residual",
        "grid": [asdict(h) for h in grid],
        "tune_results": results,
        "best": best,
        "epoch_budget": epoch_budget,
        "blend_w": blend_w,
        "har_tune_beta": beta_tune.tolist(),
        "har_final_beta": beta_final.tolist(),
        "scaler": {"mu": final_data["mu"], "sd": final_data["sd"], "fit_on": "train+val"},
        "test": test_metrics,
        "naive_test_same_rows": naive_metrics,
        "har_test_same_rows": har_metrics,
        "n_test_windows": int(len(y_test)),
        "beats_har_rmse": bool(test_metrics["rmse_log"] < har_metrics["rmse_log"]),
        "optimizer": "Adam",
    }
    (OUT / "lstm_stats.json").write_text(json.dumps(stats, indent=2))

    # Forecasts parquet for later compare step
    naive_series = df[LOG_RV_COL].astype(float)
    har_series = pd.Series(har_all, index=df.index, dtype=float)
    out_fc = df[["date_utc", "split", TARGET_COL, LOG_RV_COL]].copy()
    out_fc["yhat_lstm"] = yhat_series
    out_fc["yhat_naive"] = naive_series
    out_fc["yhat_har"] = har_series
    out_fc.to_parquet(OUT / "lstm_forecasts.parquet", index=False)

    # --- figures ---
    # val RMSE bar for top configs
    res_df = pd.DataFrame(results).sort_values("val_rmse")
    top = res_df.head(12).iloc[::-1]
    labels = [
        f"L{r.L}/h{r.hidden}/d{r.depth}/dr{r.dropout}/lr{r.lr:.0e}/w{r.blend_w:.2f}"
        for r in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(labels, top["val_rmse"].to_numpy(), color=C_LSTM, height=0.7)
    ax.set_xlabel("Val RMSE (log-RV)")
    ax.set_title("LSTM grid search (top 12 by val RMSE)")
    save(fig, "lstm_val_grid.png")

    # learning curve of best tune run — re-run briefly for history plot
    tune_data = prepare_scaled_series(
        raw, resid_tune, level, har_tune, best_hp.L, train_idx, train_idx, val_idx, None
    )
    _, _, _, _, hist = train_one(
        best_hp,
        n_in,
        tune_data["X_tr"],
        tune_data["y_tr"],
        tune_data["X_va"],
        tune_data["y_va"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        har_va=tune_data["har_va"],
        y_va_level=tune_data["level_va"],
    )
    hdf = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(hdf["epoch"], hdf["train_rmse"], color="#6B7280", lw=1.3, label="Train RMSE")
    ax.plot(hdf["epoch"], hdf["val_rmse"], color=C_LSTM, lw=1.5, label="Val RMSE")
    ax.axvline(epoch_budget, color="#DC2626", ls="--", lw=1.0, label=f"Chosen epoch={epoch_budget}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (log-RV)")
    ax.set_title("Best config: early stopping on val")
    ax.legend(frameon=False)
    save(fig, "lstm_learning_curve.png")

    test_mask = df["split"].eq("test") & yhat_series.notna()
    test_start = df.loc[test_mask, "date_utc"].iloc[0]
    plot_2mo = test_mask & (df["date_utc"] >= test_start) & (
        df["date_utc"] < test_start + pd.DateOffset(months=2)
    )
    d = df.loc[plot_2mo]
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(d["date_utc"], d[TARGET_COL], color=C_ACTUAL, lw=1.35, label="Actual $y$")
    ax.plot(d["date_utc"], naive_series.loc[d.index], color=C_NAIVE, lw=1.1, label="Naive", alpha=0.9)
    ax.plot(d["date_utc"], har_series.loc[d.index], color=C_HAR, lw=1.15, label="HAR")
    ax.plot(d["date_utc"], yhat_series.loc[d.index], color=C_LSTM, lw=1.25, label="LSTM")
    ax.set_ylabel(r"$\log\mathrm{RV}$")
    ax.set_title("Naive / HAR / LSTM vs actual (first two test months)")
    ax.set_xlim(d["date_utc"].iloc[0], d["date_utc"].iloc[-1])
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    save(fig, "lstm_test_paths_2mo.png")

    plot_2026 = test_mask & (df["date_utc"] >= pd.Timestamp("2026-01-01", tz="UTC"))
    d2 = df.loc[plot_2026]
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(d2["date_utc"], d2[TARGET_COL], color=C_ACTUAL, lw=1.35, label="Actual $y$")
    ax.plot(d2["date_utc"], naive_series.loc[d2.index], color=C_NAIVE, lw=1.1, label="Naive", alpha=0.9)
    ax.plot(d2["date_utc"], har_series.loc[d2.index], color=C_HAR, lw=1.15, label="HAR")
    ax.plot(d2["date_utc"], yhat_series.loc[d2.index], color=C_LSTM, lw=1.25, label="LSTM")
    ax.set_ylabel(r"$\log\mathrm{RV}$")
    ax.set_title("Naive / HAR / LSTM vs actual (2026)")
    ax.set_xlim(d2["date_utc"].iloc[0], d2["date_utc"].iloc[-1])
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    save(fig, "lstm_test_paths_2026.png")

    # metric bars vs Naive / HAR context
    models = ["Naive", "HAR", "LSTM"]
    colors = [C_NAIVE, C_HAR, C_LSTM]
    boards = [naive_metrics, har_metrics, test_metrics]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, key, title in zip(
        axes,
        ["rmse_log", "mae_log", "qlike"],
        ["RMSE (log-RV)", "MAE (log-RV)", "QLIKE"],
    ):
        ax.bar(models, [b[key] for b in boards], color=colors, width=0.65)
        ax.set_title(title)
    fig.suptitle("Test metrics (same rows)", y=1.02, fontsize=13, fontweight="bold")
    save(fig, "lstm_test_metrics.png")

    print(json.dumps({"best": best, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
