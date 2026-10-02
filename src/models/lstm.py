"""Standalone LSTM under eval_freeze Protocol B.

The forecast is the network output only.
Input: length-L window of log RV, its 7-day and 22-day means, squared return, and volume z-score.
Tune on train with early stopping on val, refit train+val, score test once.
"""

from __future__ import annotations

from paths import FIGURES, FORECASTS, STATS, SPLITS_DIR, ensure_result_dirs
from splits.protocol import (
    LOG_RV_COL,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    add_split_column,
    fit_mask,
    load_model_ready,
    naive_log_rv_forecast,
    split_summaries,
)
from metrics.scores import (
    evaluate_split,
    metrics_from_log_forecast,
    metrics_from_sigma_forecast,
)

FIG = FIGURES
SPLITS = SPLITS_DIR


import json
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
CHANNELS = ["log_rv", "log_rv_w", "log_rv_m", "ret_d_sq", "volume_z30"]
HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]

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


def scale_channels(raw: np.ndarray, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = raw[fit_idx].mean(axis=0)
    sd = raw[fit_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32), mu, sd


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
) -> tuple[VolLSTM, int, float, list[dict]]:
    set_seed(SEED)
    model = VolLSTM(hp, n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = nn.MSELoss()
    history: list[dict] = []
    best_state = None
    best_val = np.inf
    best_epoch = 0
    bad = 0
    tr_loader = loader(X_tr, y_tr, shuffle=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        train_rmse = float(np.sqrt(total / max(n, 1)))
        row: dict = {"epoch": epoch, "train_rmse": train_rmse}
        if X_va is not None and y_va is not None and len(X_va):
            val_rmse = float(np.sqrt(np.mean((predict(model, X_va) - y_va) ** 2)))
            row["val_rmse"] = val_rmse
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_epoch = epoch
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
    return model, best_epoch, float(best_val), history


def prepare(
    raw: np.ndarray,
    y: np.ndarray,
    L: int,
    fit_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
) -> dict:
    scaled, mu, sd = scale_channels(raw, fit_idx)
    X_tr, y_tr, i_tr = make_windows(scaled, y, L, train_idx)
    out: dict = {"mu": mu.tolist(), "sd": sd.tolist(), "X_tr": X_tr, "y_tr": y_tr, "i_tr": i_tr}
    if val_idx is not None:
        X_va, y_va, i_va = make_windows(scaled, y, L, val_idx)
        out.update({"X_va": X_va, "y_va": y_va, "i_va": i_va})
    if test_idx is not None:
        X_te, y_te, i_te = make_windows(scaled, y, L, test_idx)
        out.update({"X_te": X_te, "y_te": y_te, "i_te": i_te})
    all_idx = np.arange(L - 1, len(y))
    X_all, y_all, i_all = make_windows(scaled, y, L, all_idx)
    out.update({"X_all": X_all, "y_all": y_all, "i_all": i_all})
    return out


def fit_har(df: pd.DataFrame, fit_idx: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(fit_idx)), df.iloc[fit_idx][HAR_COLS].to_numpy(dtype=float)])
    y = df.iloc[fit_idx][TARGET_COL].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    X_all = np.column_stack([np.ones(len(df)), df[HAR_COLS].to_numpy(dtype=float)])
    return X_all @ beta


def run() -> None:
    ensure_result_dirs()
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

    har_final = fit_har(df, fit_idx)
    har_val_rmse = float(np.sqrt(np.mean((har_final[val_idx] - y[val_idx]) ** 2)))
    # HAR for the comparison table is fit on train+val, same as the baselines report.
    # Validation RMSE below uses that same fit only as a reference print, not as an LSTM input.

    grid = [
        HParams(L=L, hidden=h, depth=d, dropout=drop, lr=lr)
        for L, h, d, drop, lr in product(GRID_L, GRID_HIDDEN, GRID_DEPTH, GRID_DROPOUT, GRID_LR)
    ]
    print(f"grid size={len(grid)} channels={CHANNELS}")
    results: list[dict] = []
    best: dict | None = None

    for k, hp in enumerate(grid, 1):
        data = prepare(raw, y, hp.L, train_idx, train_idx, val_idx, None)
        _model, best_epoch, best_val, hist = train_one(
            hp, n_in, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"],
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        row = {
            **asdict(hp),
            "best_epoch": best_epoch,
            "val_rmse": best_val,
            "n_train": int(len(data["y_tr"])),
            "n_val": int(len(data["y_va"])),
            "epochs_ran": len(hist),
        }
        results.append(row)
        print(
            f"[{k:02d}/{len(grid)}] L={hp.L} h={hp.hidden} d={hp.depth} "
            f"lr={hp.lr:.0e} → val_rmse={best_val:.4f} epoch={best_epoch}"
        )
        if best is None or best_val < best["val_rmse"]:
            best = {**row}

    assert best is not None
    # Average the validation-best networks if that average is better on val than the single best.
    ranked = sorted(results, key=lambda r: r["val_rmse"])
    candidates = ranked[:3]
    val_hats = []
    for row in candidates:
        hp = HParams(row["L"], row["hidden"], row["depth"], row["dropout"], row["lr"])
        data = prepare(raw, y, hp.L, train_idx, train_idx, val_idx, None)
        model, _, _, _ = train_one(
            hp, n_in, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"],
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        pred = np.full(len(df), np.nan)
        pred[data["i_va"]] = predict(model, data["X_va"])
        val_hats.append(pred[val_idx])
    ens_val = np.mean(np.vstack(val_hats), axis=0)
    ens_val_rmse = float(np.sqrt(np.mean((ens_val - y[val_idx]) ** 2)))
    use_ensemble = ens_val_rmse + 1e-6 < float(best["val_rmse"])
    chosen = candidates if use_ensemble else [best]
    print(f"best single val={best['val_rmse']:.4f} ensemble3 val={ens_val_rmse:.4f} use_ensemble={use_ensemble}")
    print(f"HAR reference val rmse (train+val fit, not used in LSTM)={har_val_rmse:.4f}")

    test_hats = []
    full_hats = []
    histories = []
    for row in chosen:
        hp = HParams(int(row["L"]), int(row["hidden"]), int(row["depth"]), float(row["dropout"]), float(row["lr"]))
        epoch_budget = max(int(row["best_epoch"]), 1)
        data = prepare(raw, y, hp.L, fit_idx, fit_idx, None, test_idx)
        model, _, _, hist = train_one(
            hp, n_in, data["X_tr"], data["y_tr"], None, None,
            max_epochs=epoch_budget, patience=None,
        )
        histories.append(hist)
        te = np.full(len(df), np.nan)
        te[data["i_te"]] = predict(model, data["X_te"])
        test_hats.append(te)
        full = np.full(len(df), np.nan)
        full[data["i_all"]] = predict(model, data["X_all"])
        full_hats.append(full)

    yhat = np.nanmean(np.vstack(test_hats), axis=0)
    yhat_all = np.nanmean(np.vstack(full_hats), axis=0)
    test_rows = np.where(np.isin(np.arange(len(df)), test_idx) & np.isfinite(yhat))[0]
    y_test = y[test_rows]
    yhat_test = yhat[test_rows]
    test_metrics = metrics_from_log_forecast(y_test, yhat_test)
    naive_metrics = metrics_from_log_forecast(y_test, df.iloc[test_rows][LOG_RV_COL].to_numpy(dtype=float))
    har_metrics = metrics_from_log_forecast(y_test, har_final[test_rows])

    yhat_series = pd.Series(yhat_all, index=df.index, dtype=float)
    stats = {
        "seed": SEED,
        "channels": CHANNELS,
        "model": "LSTM",
        "grid": [asdict(h) for h in grid],
        "tune_results": results,
        "best_single": best,
        "selected": chosen,
        "use_ensemble": use_ensemble,
        "ensemble_val_rmse": ens_val_rmse,
        "test": test_metrics,
        "naive_test_same_rows": naive_metrics,
        "har_test_same_rows": har_metrics,
        "beats_har_rmse": bool(test_metrics["rmse_log"] < har_metrics["rmse_log"]),
        "optimizer": "Adam",
    }
    (STATS / "lstm_stats.json").write_text(json.dumps(stats, indent=2))

    naive_series = df[LOG_RV_COL].astype(float)
    har_series = pd.Series(har_final, index=df.index, dtype=float)
    out_fc = df[["date_utc", "split", TARGET_COL, LOG_RV_COL]].copy()
    out_fc["yhat_lstm"] = yhat_series
    out_fc["yhat_naive"] = naive_series
    out_fc["yhat_har"] = har_series
    out_fc.to_parquet(FORECASTS / "lstm_forecasts.parquet", index=False)

    res_df = pd.DataFrame(results).sort_values("val_rmse")
    top = res_df.head(12).iloc[::-1]
    labels = [f"L{r.L}/h{r.hidden}/d{r.depth}/lr{r.lr:.0e}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(labels, top["val_rmse"].to_numpy(), color=C_LSTM, height=0.7)
    ax.set_xlabel("Val RMSE (log-RV)")
    ax.set_title("LSTM validation grid")
    save(fig, "lstm_val_grid.png")

    hp = HParams(int(best["L"]), int(best["hidden"]), int(best["depth"]), float(best["dropout"]), float(best["lr"]))
    tune_data = prepare(raw, y, hp.L, train_idx, train_idx, val_idx, None)
    _, epoch_budget, _, hist = train_one(
        hp, n_in, tune_data["X_tr"], tune_data["y_tr"], tune_data["X_va"], tune_data["y_va"],
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
    )
    hdf = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(hdf["epoch"], hdf["train_rmse"], color="#6B7280", lw=1.3, label="Train RMSE")
    ax.plot(hdf["epoch"], hdf["val_rmse"], color=C_LSTM, lw=1.5, label="Val RMSE")
    ax.axvline(int(best["best_epoch"]), color="#DC2626", ls="--", lw=1.0, label=f"Chosen epoch={int(best['best_epoch'])}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (log-RV)")
    ax.set_title("Validation curve for the selected specification")
    ax.legend(frameon=False)
    save(fig, "lstm_learning_curve.png")

    test_mask = df["split"].eq("test") & yhat_series.notna()
    test_start = df.loc[test_mask, "date_utc"].iloc[0]
    windows = [
        (
            test_mask & (df["date_utc"] >= test_start) & (df["date_utc"] < test_start + pd.DateOffset(months=2)),
            "Naive / HAR / LSTM (first two test months)",
            "lstm_test_paths_2mo.png",
        ),
        (
            test_mask & (df["date_utc"] >= pd.Timestamp("2026-01-01", tz="UTC")),
            "Naive / HAR / LSTM (2026)",
            "lstm_test_paths_2026.png",
        ),
    ]
    for mask, title, name in windows:
        d = df.loc[mask]
        fig, ax = plt.subplots(figsize=(10.5, 4.0))
        ax.plot(d["date_utc"], d[TARGET_COL], color=C_ACTUAL, lw=1.35, label="Actual $y$")
        ax.plot(d["date_utc"], naive_series.loc[d.index], color=C_NAIVE, lw=1.1, label="Naive", alpha=0.9)
        ax.plot(d["date_utc"], har_series.loc[d.index], color=C_HAR, lw=1.15, label="HAR")
        ax.plot(d["date_utc"], yhat_series.loc[d.index], color=C_LSTM, lw=1.25, label="LSTM")
        ax.set_ylabel(r"$\log\mathrm{RV}$")
        ax.set_title(title)
        ax.set_xlim(d["date_utc"].iloc[0], d["date_utc"].iloc[-1])
        ax.legend(frameon=False, ncol=4, fontsize=9)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        save(fig, name)

    boards = [naive_metrics, har_metrics, test_metrics]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6))
    for ax, key, title in zip(
        axes,
        ["rmse_log", "mae_log", "mape_rv", "qlike"],
        ["RMSE (log-RV)", "MAE (log-RV)", "MAPE (RV, %)", "QLIKE"],
    ):
        ax.bar(["Naive", "HAR", "LSTM"], [b[key] for b in boards], color=[C_NAIVE, C_HAR, C_LSTM], width=0.65)
        ax.set_title(title)
    fig.suptitle("Test metrics", y=1.02, fontsize=13, fontweight="bold")
    save(fig, "lstm_test_metrics.png")

    print(json.dumps({
        "use_ensemble": use_ensemble,
        "test": test_metrics,
        "har": har_metrics,
        "beats_har_rmse": stats["beats_har_rmse"],
    }, indent=2))


if __name__ == "__main__":
    run()
