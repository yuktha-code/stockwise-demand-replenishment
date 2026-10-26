"""StockWise: independent M5 forecasting and replenishment experiment.

No upstream project implementation is copied. See README.md and DATA_SOURCES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SOURCE = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
SEED = 20261016
MODELS = ("seasonal_naive_7", "rolling_mean_28", "poisson_boosting")
FEATURES = ["lag1", "lag7", "lag14", "lag28", "mean7", "mean28", "std28",
            "dow", "month", "dayofmonth", "series_code", "store_code", "dept_code"]


def load_data(archive: Path, out: Path, n_series: int = 50):
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as z:
        with z.open("sales_train_evaluation.csv") as f:
            # Selection uses identifiers and the first 1,000 days only.
            cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
            early = pd.read_csv(f, usecols=cols + [f"d_{i}" for i in range(1, 1001)])
        early["id"] = early.item_id + "_" + early.store_id
        eligible = early.loc[early[[f"d_{i}" for i in range(1, 1001)]].sum(axis=1) >= 100, "id"].sort_values()
        rng = np.random.default_rng(SEED)
        selected = sorted(rng.choice(eligible.to_numpy(), size=n_series, replace=False).tolist())
        with z.open("sales_train_evaluation.csv") as f:
            sales = pd.read_csv(f)
        sales["id"] = sales.item_id + "_" + sales.store_id
        sales = sales.set_index("id").loc[selected].reset_index()
        with z.open("calendar.csv") as f:
            cal = pd.read_csv(f)
    dcols = sorted([c for c in sales if c.startswith("d_")], key=lambda x: int(x[2:]))
    y = sales[dcols].to_numpy(dtype=float)
    # Nixtla's mirror omits the original redundant d_N calendar column and ID.
    dates = pd.to_datetime(cal["date"].iloc[:len(dcols)]).reset_index(drop=True)
    static = pd.DataFrame({
        "series_code": np.arange(n_series),
        "store_code": pd.Categorical(sales.store_id).codes,
        "dept_code": pd.Categorical(sales.dept_id).codes,
    }).to_numpy()
    assert len(selected) == len(set(selected)) and np.isfinite(y).all() and (y >= 0).all()
    assert dates.is_unique and (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    manifest = {
        "source_url": SOURCE, "archive_sha256": digest, "selection_seed": SEED,
        "selection_rule": "50 IDs sampled from series with >=100 units in first 1000 days; no validation/test filtering",
        "series_ids": selected, "n_series": n_series, "days_per_series": y.shape[1],
        "start_date": str(dates.iloc[0].date()), "end_date": str(dates.iloc[-1].date()),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "features": FEATURES,
        "limitations": ["Recorded retail sales proxy for demand; unobserved lost sales not recovered",
                        "Small sampled subset, not official M5 leaderboard evaluation",
                        "Lead times, inventory states and costs are simulated assumptions"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    return y, dates, static, sales[["id"] + cols], manifest


def feature_rows(history: np.ndarray, date: pd.Timestamp, static: np.ndarray) -> np.ndarray:
    """Only sales strictly before the forecasted date are accepted here."""
    n = history.shape[0]
    return np.column_stack([
        history[:, -1], history[:, -7], history[:, -14], history[:, -28],
        history[:, -7:].mean(axis=1), history[:, -28:].mean(axis=1),
        history[:, -28:].std(axis=1), np.full(n, date.dayofweek),
        np.full(n, date.month), np.full(n, date.day), static,
    ])


def fit_model(y, dates, static, cutoff):
    start = max(28, cutoff - 730)
    x = np.concatenate([feature_rows(y[:, :t], dates.iloc[t], static) for t in range(start, cutoff)])
    target = np.concatenate([y[:, t] for t in range(start, cutoff)])
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=120, max_leaf_nodes=15, min_samples_leaf=30,
        learning_rate=0.08, l2_regularization=1.0, early_stopping=False,
        categorical_features=[10, 11, 12], random_state=SEED,
    )
    model.fit(x, target)
    return model


def predict_horizon(history, dates, start, horizon, static, model, method):
    if method == "seasonal_naive_7":
        return np.column_stack([history[:, -7 + (j % 7)] for j in range(horizon)])
    if method == "rolling_mean_28":
        return np.repeat(history[:, -28:].mean(axis=1)[:, None], horizon, axis=1)
    h = history.copy()
    predictions = []
    for j in range(horizon):
        pred = np.maximum(0, model.predict(feature_rows(h, dates.iloc[start + j], static)))
        predictions.append(pred)
        h = np.column_stack([h, pred])
    return np.column_stack(predictions)


def forecast_metrics(actual, pred):
    denom = actual.sum()
    return {"mae": float(np.abs(actual - pred).mean()),
            "wape": float(np.abs(actual - pred).sum() / denom),
            "bias": float((pred - actual).sum() / denom)}


def run_forecasts(y, dates, static, ids, out):
    horizon = 28
    # Three pre-test windows. Last 28 days stay untouched until final scoring.
    cutoffs = [y.shape[1] - horizon * k for k in (4, 3, 2, 1)]
    summaries, rows = [], []
    final_model = None
    for fold, cutoff in enumerate(cutoffs):
        split = "test" if fold == 3 else "validation"
        print(f"Fitting {split} window {fold + 1}, cutoff {dates.iloc[cutoff - 1].date()}", flush=True)
        model = fit_model(y, dates, static, cutoff)
        for method in MODELS:
            pred = predict_horizon(y[:, :cutoff], dates, cutoff, horizon, static, model, method)
            actual = y[:, cutoff:cutoff + horizon]
            summaries.append({"split": split, "fold": fold + 1, "method": method,
                              "cutoff": str(dates.iloc[cutoff - 1].date()), **forecast_metrics(actual, pred)})
            for i, uid in enumerate(ids):
                rows.extend({"split": split, "fold": fold + 1, "method": method, "series_id": uid,
                             "date": str(dates.iloc[cutoff + j].date()), "actual": float(actual[i, j]),
                             "forecast": float(pred[i, j])} for j in range(horizon))
        if split == "test":
            final_model = model
    metrics = pd.DataFrame(summaries)
    forecasts = pd.DataFrame(rows)
    assert not forecasts.duplicated(["split", "fold", "method", "series_id", "date"]).any()
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    return metrics, forecasts, final_model, cutoffs[-1]


def simulate_inventory(y, dates, static, ids, model, cutoff, lead_time):
    """Daily periodic-review, lost-sales simulation. All business costs assumed.

    Order placed at t arrives at start of t+lead_time. Protection interval L+1.
    Day t demand is not visible when forecasting or ordering at t.
    Orders use stock position after arrivals, before today's sales.
    Same initial stock, no initial pipeline, same fixed safety stock for all methods.
    """
    horizon, n = 28, len(ids)
    protection = lead_time + 1
    train_recent = y[:, cutoff - 56:cutoff]
    mean = train_recent.mean(axis=1)
    std = train_recent.std(axis=1)
    # A common heuristic buffer, not a demonstrated 95% service guarantee.
    safety = np.ceil(1.645 * std * np.sqrt(protection))
    initial = np.ceil(mean * protection + safety)
    output = []
    # Extend known calendar only; future sales are never added here.
    future_dates = pd.Series(pd.date_range(dates.iloc[0], periods=len(dates) + protection, freq="D"))
    for method in MODELS:
        on_hand = initial.copy()
        pipeline = np.zeros((horizon + protection + 1, n))
        for day in range(horizon):
            t = cutoff + day
            on_hand += pipeline[day]
            position = on_hand + pipeline[day + 1:].sum(axis=0)
            pred = predict_horizon(y[:, :t], future_dates, t, protection, static, model, method)
            target = np.ceil(pred.sum(axis=1) + safety)
            order = np.maximum(0, target - position)
            pipeline[day + lead_time] += order
            demand = y[:, t]
            fulfilled = np.minimum(on_hand, demand)
            lost = demand - fulfilled
            on_hand -= fulfilled
            # Unitless scenario costs: holding=1/unit/day, lost-sales penalty=10/unit.
            cost = on_hand + 10 * lost
            for i, uid in enumerate(ids):
                output.append({"method": method, "lead_time": lead_time, "series_id": uid,
                               "date": str(dates.iloc[t].date()), "sales_proxy": float(demand[i]),
                               "fulfilled": float(fulfilled[i]), "lost_sales": float(lost[i]),
                               "on_hand": float(on_hand[i]), "ordered": float(order[i]),
                               "open_orders": float(pipeline[day + 1:, i].sum()),
                               "assumed_cost": float(cost[i])})
    return pd.DataFrame(output)


def save_inventory_and_sql(frames, forecasts, metrics, metadata, out):
    inv = pd.concat(frames, ignore_index=True)
    assert not inv.duplicated(["method", "lead_time", "series_id", "date"]).any()
    assert (inv[["fulfilled", "lost_sales", "on_hand", "ordered", "open_orders"]] >= 0).all().all()
    assert np.allclose(inv.fulfilled + inv.lost_sales, inv.sales_proxy)
    inv.to_csv(out / "inventory_daily.csv", index=False)
    db = out / "stockwise.sqlite"
    if db.exists():
        db.unlink()
    sql = (Path(__file__).parent / "report_queries.sql").read_text()
    with sqlite3.connect(db) as conn:
        forecasts.to_sql("forecasts", conn, index=False)
        inv.to_sql("inventory_daily", conn, index=False)
        metrics.to_sql("forecast_metrics", conn, index=False)
        metadata.to_sql("series_metadata", conn, index=False)
        conn.executescript(sql)
        summary = pd.read_sql_query("SELECT * FROM inventory_summary ORDER BY lead_time, method", conn)
        exceptions = pd.read_sql_query("SELECT * FROM sku_risk ORDER BY lost_sales DESC LIMIT 25", conn)
    summary.to_csv(out / "inventory_summary.csv", index=False)
    exceptions.to_csv(out / "sku_exceptions.csv", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=Path("data/m5.zip"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        y, dates, static, meta, manifest = load_data(args.archive, args.out)
        metrics, forecasts, model, cutoff = run_forecasts(y, dates, static, meta.id, args.out)
        frames = [simulate_inventory(y, dates, static, meta.id, model, cutoff, lead) for lead in (3, 7)]
        inv_summary = save_inventory_and_sql(frames, forecasts, metrics, meta, args.out)
    cv = metrics.query("split == 'validation'").groupby("method").wape.mean()
    test = metrics.query("split == 'test'").set_index("method")
    report = {
        "n_series": len(y), "forecast_horizon_days": 28, "validation_windows": 3,
        "test_start": str(dates.iloc[cutoff].date()), "test_end": str(dates.iloc[-1].date()),
        "model_hyperparameters_fixed_before_evaluation": True,
        "validation_mean_wape": cv.to_dict(), "validation_selected_method": cv.idxmin(),
        "test_metrics": test[["mae", "wape", "bias"]].to_dict(orient="index"),
        "inventory_lead_times_days": [3, 7], "inventory_is_simulated": True,
        "archive_sha256": manifest["archive_sha256"],
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(inv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
