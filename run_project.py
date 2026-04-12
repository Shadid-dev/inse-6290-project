
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "date"]).copy()
    grp = df.groupby("item_id")
    df["lag_1"]     = grp["demand"].shift(1)
    df["lag_7"]     = grp["demand"].shift(7)
    df["rolling_3"] = grp["demand"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["rolling_7"] = grp["demand"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    return df


# ─────────────────────────────────────────────
# Model training & forecasting
# ─────────────────────────────────────────────
def train_and_forecast(df: pd.DataFrame):
    df = prepare_features(df).dropna().copy()
    cutoff = df["date"].sort_values().unique()[-14]
    train  = df[df["date"] <  cutoff].copy()
    test   = df[df["date"] >= cutoff].copy()

    feature_cols = ["item_id", "day_of_week", "is_weekend",
                    "lag_1", "lag_7", "rolling_3", "rolling_7"]
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=10, min_samples_leaf=2, random_state=42
    )
    rf.fit(train[feature_cols], train["demand"])
    test["rf_forecast"]       = np.maximum(0, rf.predict(test[feature_cols]))
    test["baseline_forecast"] = np.maximum(0, test["lag_7"])

    metrics = {
        "Baseline_MAE":  float(mean_absolute_error(test["demand"], test["baseline_forecast"])),
        "Baseline_RMSE": float(np.sqrt(mean_squared_error(test["demand"], test["baseline_forecast"]))),
        "RF_MAE":        float(mean_absolute_error(test["demand"], test["rf_forecast"])),
        "RF_RMSE":       float(np.sqrt(mean_squared_error(test["demand"], test["rf_forecast"]))),
    }
    residual_std = train.groupby("item_id")["demand"].std().to_dict()
    item_params  = (
        train.groupby(["item_id", "item_name", "holding_cost",
                       "shortage_cost", "unit_cost", "selling_price", "shelf_life_days"])
        .size().reset_index().drop(columns=0)
    )
    feat_imp = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).round(4)

    return train, test, metrics, residual_std, item_params, feat_imp


# ─────────────────────────────────────────────
# Newsvendor critical-ratio Z-score lookup
# ─────────────────────────────────────────────
def normal_z_from_service(service_level: float) -> float:
    lookup = {
        0.50: 0.0,   0.60: 0.253, 0.65: 0.385, 0.70: 0.524, 0.75: 0.674,
        0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.91: 1.341, 0.92: 1.405,
        0.93: 1.476, 0.94: 1.555, 0.95: 1.645, 0.96: 1.751, 0.97: 1.881,
        0.98: 2.054, 0.99: 2.326
    }
    keys = np.array(sorted(lookup.keys()))
    idx  = np.abs(keys - service_level).argmin()
    return lookup[float(keys[idx])]


# ─────────────────────────────────────────────
# Inventory simulation (FEFO age-bucket model)
# ─────────────────────────────────────────────
def simulate_inventory(test: pd.DataFrame, item_params: pd.DataFrame,
                       residual_std: dict, forecast_col: str) -> pd.DataFrame:
    params = item_params.set_index("item_id").to_dict(orient="index")
    records = []
    state   = {}
    for item_id, p in params.items():
        life = int(p["shelf_life_days"])
        state[item_id] = {"buckets": [0] * life}

    for date in sorted(test["date"].unique()):
        day = test[test["date"] == date].copy()
        for _, row in day.iterrows():
            item_id  = int(row["item_id"])
            p        = params[item_id]
            holding  = float(p["holding_cost"])
            shortage = float(p["shortage_cost"])
            crit_ratio = shortage / (shortage + holding)
            z        = normal_z_from_service(round(min(max(crit_ratio, 0.5), 0.99), 2))
            sigma    = residual_std.get(item_id, 4.0)
            mu       = max(0.0, float(row[forecast_col]))
            order_qty = int(round(max(0, mu + z * sigma)))

            buckets  = state[item_id]["buckets"]
            waste    = buckets[0]                       # oldest bucket expires
            buckets  = buckets[1:] + [order_qty]       # shift age; receive order
            available = sum(buckets)
            demand   = int(row["demand"])
            sold     = min(available, demand)
            remaining = sold
            for i in range(len(buckets)):               # FEFO: consume oldest first
                take       = min(buckets[i], remaining)
                buckets[i] -= take
                remaining  -= take
            end_inventory = sum(buckets)
            stockout = max(0, demand - sold)

            state[item_id]["buckets"] = buckets
            records.append({
                "date":             date,
                "item_id":          item_id,
                "item_name":        row["item_name"],
                "forecast_method":  forecast_col,
                "forecast":         mu,
                "actual_demand":    demand,
                "order_qty":        order_qty,
                "sales":            sold,
                "stockout_units":   stockout,
                "waste_units":      int(waste),
                "ending_inventory": int(end_inventory),
                "unit_cost":        float(p["unit_cost"]),
                "selling_price":    float(p["selling_price"])
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Supply-chain KPI aggregation
# ─────────────────────────────────────────────
def compute_supply_chain_metrics(sim_df: pd.DataFrame) -> pd.Series:
    total_demand  = sim_df["actual_demand"].sum()
    total_sales   = sim_df["sales"].sum()
    total_orders  = sim_df["order_qty"].sum()
    total_waste   = sim_df["waste_units"].sum()
    avg_inventory = max(1e-6, sim_df["ending_inventory"].mean())
    return pd.Series({
        "Total Demand":      total_demand,
        "Total Sales":       total_sales,
        "Total Ordered":     total_orders,
        "Fill Rate":         round(total_sales / total_demand if total_demand else 0, 4),
        "Stockout Rate":     round(sim_df["stockout_units"].sum() / total_demand if total_demand else 0, 4),
        "Food Waste %":      round(total_waste / total_orders if total_orders else 0, 4),
        "Inventory Turnover": round(total_sales / avg_inventory, 2)
    })


# ─────────────────────────────────────────────
# Static plots saved to disk
# ─────────────────────────────────────────────
def make_plots(train, test, sim_baseline, sim_rf, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Sample item forecast comparison
    item   = "Burger"
    sample = test[test["item_name"] == item].copy()
    plt.figure(figsize=(9, 4.8))
    plt.plot(sample["date"], sample["demand"],            marker="o", label="Actual demand")
    plt.plot(sample["date"], sample["baseline_forecast"], marker="o", label="Baseline (Lag-7)")
    plt.plot(sample["date"], sample["rf_forecast"],       marker="o", label="AI (Random Forest)")
    plt.title(f"{item}: Actual vs Forecasted Daily Demand (Test Horizon)")
    plt.xlabel("Date"); plt.ylabel("Units")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.legend()
    plt.savefig(outdir / "forecast_comparison_burger.png", dpi=200)
    plt.close()

    # Plot 2: Daily supply-chain outcomes per policy
    for df, name in [(sim_baseline, "baseline"), (sim_rf, "rf")]:
        daily = df.groupby("date")[["sales", "stockout_units", "waste_units"]].sum()
        plt.figure(figsize=(9, 4.8))
        plt.plot(daily.index, daily["sales"],          marker="o", label="Sales")
        plt.plot(daily.index, daily["stockout_units"], marker="o", label="Stockouts")
        plt.plot(daily.index, daily["waste_units"],    marker="o", label="Waste")
        plt.title(f"Daily Supply Chain Outcomes – {name.upper()} policy")
        plt.xlabel("Date"); plt.ylabel("Units")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.legend()
        plt.savefig(outdir / f"daily_outcomes_{name}.png", dpi=200)
        plt.close()

    # Plot 3: Aggregate training demand trend
    plt.figure(figsize=(8.6, 4.8))
    train.groupby("date")["demand"].sum().plot()
    plt.title("Aggregate Daily Demand – All Menu Items (Training Period)")
    plt.xlabel("Date"); plt.ylabel("Units"); plt.tight_layout()
    plt.savefig(outdir / "aggregate_training_demand.png", dpi=200)
    plt.close()


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def main():
    # Project root = same directory as this script
    project_root = Path(__file__).resolve().parent
    data_path    = project_root / "synthetic_food_orders.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run  python generate_synthetic_data.py  first."
        )

    df     = pd.read_csv(data_path, parse_dates=["date"])
    outdir = project_root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    train, test, forecast_metrics, residual_std, item_params, feat_imp = train_and_forecast(df)
    sim_baseline = simulate_inventory(test, item_params, residual_std, "baseline_forecast")
    sim_rf       = simulate_inventory(test, item_params, residual_std, "rf_forecast")

    scm_metrics = pd.DataFrame({
        "Baseline": compute_supply_chain_metrics(sim_baseline),
        "AI-RF":    compute_supply_chain_metrics(sim_rf),
    })

    # Save outputs
    make_plots(train, test, sim_baseline, sim_rf, outdir)
    pd.DataFrame([forecast_metrics]).round(4).to_csv(outdir / "forecast_metrics.csv",    index=False)
    scm_metrics.to_csv(outdir / "supply_chain_metrics.csv")
    feat_imp.to_csv(outdir / "feature_importance.csv", index=False)
    sim_baseline.to_csv(outdir / "simulation_baseline.csv", index=False)
    sim_rf.to_csv(outdir / "simulation_ai_rf.csv",         index=False)

    summary = {
        "forecast_metrics":      forecast_metrics,
        "supply_chain_metrics":  scm_metrics.to_dict(),
        "dataset_rows":          int(len(df)),
        "train_days":            int(train["date"].nunique()),
        "test_days":             int(test["date"].nunique())
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== Forecast Accuracy ===")
    print(pd.DataFrame([forecast_metrics]).round(4).to_string(index=False))
    print("\n=== Supply Chain KPIs ===")
    print(scm_metrics.round(4))
    print(f"\nOutputs saved to: {outdir}")


if __name__ == "__main__":
    main()
