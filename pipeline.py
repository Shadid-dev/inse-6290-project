"""
INSE 6290 – Real-Data Pipeline
================================
Uses the actual foodDemand dataset (456k rows, 145 weeks, 77 centers, 51 meals).
Aggregates top-10 meals weekly across all centers, then runs:
  - Random Forest vs Lag-4 baseline forecasting
  - Newsvendor + FEFO inventory simulation
  - Supply-chain KPI evaluation
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# Cost parameters by meal category
# (unit_cost = 40% of avg selling price — standard food COGS ratio)
# shelf_life in weeks: fresh items = 1 week, others = 2 weeks
# ─────────────────────────────────────────────────────────────
FRESH_CATEGORIES = {"Salad", "Soup", "Fish", "Seafood"}

SHELF_LIFE_WEEKS = 1   # fresh
SHELF_LIFE_SHELF = 2   # ambient / longer-life


# ─────────────────────────────────────────────────────────────
# Data loading & preparation
# ─────────────────────────────────────────────────────────────
def load_real_data(top_n_meals: int = 10) -> pd.DataFrame:
    """
    Load, merge, aggregate, and augment the real foodDemand dataset.

    Steps:
    1. Load train.csv + meal_info.csv
    2. Pick top-N meals by total historical demand
    3. Aggregate per (meal_id, week): sum orders, mean price, max promo flags
    4. Derive cost parameters from actual prices
    5. Add shelf-life by meal category
    """
    train  = pd.read_csv(PROJECT_ROOT / "foodDemand_train" / "train.csv")
    meals  = pd.read_csv(PROJECT_ROOT / "foodDemand_train" / "meal_info.csv")

    # Merge meal category info
    df = train.merge(meals, on="meal_id", how="left")

    # Pick top-N meals by total demand
    top_meals = (
        df.groupby("meal_id")["num_orders"].sum()
        .nlargest(top_n_meals).index.tolist()
    )
    df = df[df["meal_id"].isin(top_meals)].copy()

    # Aggregate across all fulfillment centers per (meal, week)
    agg = df.groupby(["meal_id", "week"]).agg(
        num_orders           = ("num_orders",            "sum"),
        checkout_price       = ("checkout_price",        "mean"),
        base_price           = ("base_price",            "mean"),
        emailer_for_promotion= ("emailer_for_promotion", "max"),
        homepage_featured    = ("homepage_featured",     "max"),
        category             = ("category",              "first"),
        cuisine              = ("cuisine",               "first"),
    ).reset_index()

    # Derived price features
    agg["price_ratio"]    = (agg["checkout_price"] / agg["base_price"]).round(4)   # <1 means discount
    agg["price_discount"] = (agg["base_price"] - agg["checkout_price"]).round(2)   # absolute discount

    # Cost parameters derived from actual prices
    agg["selling_price"]  = agg["checkout_price"].round(2)
    agg["unit_cost"]      = (agg["checkout_price"] * 0.40).round(2)   # 40% COGS
    agg["holding_cost"]   = (agg["unit_cost"] * 0.05).round(3)        # 5% of cost/week
    agg["shortage_cost"]  = (agg["checkout_price"] * 0.50).round(2)   # 50% of price = lost margin + goodwill
    agg["shelf_life_weeks"] = agg["category"].apply(
        lambda c: SHELF_LIFE_WEEKS if c in FRESH_CATEGORIES else SHELF_LIFE_SHELF
    )

    # Week-of-year proxy (week % 52) for seasonality
    agg["week_of_year"] = ((agg["week"] - 1) % 52) + 1

    return agg.sort_values(["meal_id", "week"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["meal_id", "week"]).copy()
    grp = df.groupby("meal_id")

    df["lag_1"]     = grp["num_orders"].shift(1)
    df["lag_4"]     = grp["num_orders"].shift(4)    # ~1 month ago
    df["rolling_4"] = grp["num_orders"].shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["rolling_8"] = grp["num_orders"].shift(1).rolling(8).mean().reset_index(level=0, drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# Train & forecast
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "meal_id", "week_of_year",
    "emailer_for_promotion", "homepage_featured",
    "price_ratio", "price_discount",
    "lag_1", "lag_4", "rolling_4", "rolling_8",
]

def train_and_forecast(df: pd.DataFrame, test_weeks: int = 10):
    df    = prepare_features(df).dropna().copy()
    cutoff = df["week"].max() - test_weeks + 1
    train  = df[df["week"] <  cutoff].copy()
    test   = df[df["week"] >= cutoff].copy()

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42
    )
    rf.fit(train[FEATURE_COLS], train["num_orders"])
    test["rf_forecast"]       = np.maximum(0, rf.predict(test[FEATURE_COLS]))
    test["baseline_forecast"] = np.maximum(0, test["lag_4"])   # baseline = 4-week-ago demand

    metrics = {
        "Baseline_MAE":  float(mean_absolute_error(test["num_orders"], test["baseline_forecast"])),
        "Baseline_RMSE": float(np.sqrt(mean_squared_error(test["num_orders"], test["baseline_forecast"]))),
        "RF_MAE":        float(mean_absolute_error(test["num_orders"], test["rf_forecast"])),
        "RF_RMSE":       float(np.sqrt(mean_squared_error(test["num_orders"], test["rf_forecast"]))),
    }

    # Use in-sample FORECAST ERROR std (not raw demand std) for newsvendor safety stock.
    # This correctly captures uncertainty around the model's own predictions.
    train_pred = rf.predict(train[FEATURE_COLS])
    train["rf_residual"]       = train["num_orders"] - train_pred
    train["baseline_residual"] = train["num_orders"] - train["lag_4"].fillna(train["num_orders"].mean())
    rf_residual_std       = train.groupby("meal_id")["rf_residual"].std().to_dict()
    baseline_residual_std = train.groupby("meal_id")["baseline_residual"].std().to_dict()
    residual_std = {"rf": rf_residual_std, "baseline": baseline_residual_std}

    item_params = (
        train.groupby([
            "meal_id", "category", "cuisine",
            "unit_cost", "selling_price",
            "holding_cost", "shortage_cost", "shelf_life_weeks"
        ]).size().reset_index().drop(columns=0)
    )
    # Use average cost params per meal (prices vary week to week)
    item_params = (
        train.groupby("meal_id").agg(
            category      = ("category",       "first"),
            cuisine       = ("cuisine",        "first"),
            unit_cost     = ("unit_cost",      "mean"),
            selling_price = ("selling_price",  "mean"),
            holding_cost  = ("holding_cost",   "mean"),
            shortage_cost = ("shortage_cost",  "mean"),
            shelf_life_weeks = ("shelf_life_weeks", "first"),
        ).reset_index()
    )

    feat_imp = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False).round(4)

    return train, test, metrics, residual_std, item_params, feat_imp


# ─────────────────────────────────────────────────────────────
# Newsvendor Z-score lookup (no SciPy needed)
# ─────────────────────────────────────────────────────────────
def normal_z_from_service(service_level: float) -> float:
    lookup = {
        0.50: 0.0,   0.60: 0.253, 0.65: 0.385, 0.70: 0.524, 0.75: 0.674,
        0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.91: 1.341, 0.92: 1.405,
        0.93: 1.476, 0.94: 1.555, 0.95: 1.645, 0.96: 1.751, 0.97: 1.881,
        0.98: 2.054, 0.99: 2.326,
    }
    keys = np.array(sorted(lookup.keys()))
    idx  = np.abs(keys - service_level).argmin()
    return lookup[float(keys[idx])]


# ─────────────────────────────────────────────────────────────
# Inventory simulation — FEFO age-bucket model (weekly periods)
# ─────────────────────────────────────────────────────────────
def simulate_inventory(
    test: pd.DataFrame,
    item_params: pd.DataFrame,
    residual_std: dict,
    forecast_col: str,
) -> pd.DataFrame:
    # residual_std is a dict with keys "rf" and "baseline"
    policy_key = "rf" if "rf" in forecast_col else "baseline"
    std_lookup = residual_std[policy_key]

    params  = item_params.set_index("meal_id").to_dict(orient="index")
    records = []
    state   = {}

    for meal_id, p in params.items():
        life = int(round(p["shelf_life_weeks"]))
        state[meal_id] = {"buckets": [0] * life}

    for week in sorted(test["week"].unique()):
        week_data = test[test["week"] == week].copy()
        for _, row in week_data.iterrows():
            meal_id  = int(row["meal_id"])
            p        = params[meal_id]
            holding  = float(p["holding_cost"])
            shortage = float(p["shortage_cost"])

            crit_ratio = shortage / (shortage + holding)
            z          = normal_z_from_service(round(min(max(crit_ratio, 0.5), 0.99), 2))
            sigma      = std_lookup.get(meal_id, 100.0)
            mu         = max(0.0, float(row[forecast_col]))
            order_qty  = int(round(max(0, mu + z * sigma)))

            buckets   = state[meal_id]["buckets"]
            waste     = buckets[0]                      # oldest bucket expires
            buckets   = buckets[1:] + [order_qty]       # shift age; receive new order
            available = sum(buckets)
            demand    = int(row["num_orders"])
            sold      = min(available, demand)
            remaining = sold
            for i in range(len(buckets)):               # FEFO: consume oldest first
                take       = min(buckets[i], remaining)
                buckets[i] -= take
                remaining  -= take
            end_inv  = sum(buckets)
            stockout = max(0, demand - sold)

            state[meal_id]["buckets"] = buckets
            records.append({
                "week":             week,
                "meal_id":          meal_id,
                "category":         row["category"],
                "cuisine":          row["cuisine"],
                "forecast_method":  forecast_col,
                "forecast":         round(mu, 1),
                "actual_demand":    demand,
                "order_qty":        order_qty,
                "sales":            sold,
                "stockout_units":   stockout,
                "waste_units":      int(waste),
                "ending_inventory": int(end_inv),
                "unit_cost":        round(float(p["unit_cost"]), 2),
                "selling_price":    round(float(p["selling_price"]), 2),
                "emailer_for_promotion": int(row["emailer_for_promotion"]),
                "homepage_featured":     int(row["homepage_featured"]),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Supply-chain KPIs
# ─────────────────────────────────────────────────────────────
def compute_supply_chain_metrics(sim_df: pd.DataFrame) -> pd.Series:
    total_demand  = sim_df["actual_demand"].sum()
    total_sales   = sim_df["sales"].sum()
    total_orders  = sim_df["order_qty"].sum()
    total_waste   = sim_df["waste_units"].sum()
    avg_inventory = max(1e-6, sim_df["ending_inventory"].mean())
    revenue       = (sim_df["sales"] * sim_df["selling_price"]).sum()
    cogs          = (sim_df["order_qty"] * sim_df["unit_cost"]).sum()
    waste_cost    = (sim_df["waste_units"] * sim_df["unit_cost"]).sum()
    return pd.Series({
        "Total Demand":        int(total_demand),
        "Total Sales":         int(total_sales),
        "Total Ordered":       int(total_orders),
        "Fill Rate":           round(total_sales / total_demand if total_demand else 0, 4),
        "Stockout Rate":       round(sim_df["stockout_units"].sum() / total_demand if total_demand else 0, 4),
        "Food Waste %":        round(total_waste / total_orders if total_orders else 0, 4),
        "Inventory Turnover":  round(total_sales / avg_inventory, 2),
        "Revenue ($)":         round(revenue, 0),
        "COGS ($)":            round(cogs, 0),
        "Waste Cost ($)":      round(waste_cost, 0),
        "Gross Profit ($)":    round(revenue - cogs - waste_cost, 0),
    })


# ─────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("Loading real dataset…")
    df = load_real_data(top_n_meals=10)
    print(f"Dataset: {df.shape[0]} rows, {df['meal_id'].nunique()} meals, "
          f"weeks {df.week.min()}–{df.week.max()}")

    print("Training models…")
    train, test, fm, residual_std, item_params, feat_imp = train_and_forecast(df)

    print("Simulating inventory…")
    sim_base = simulate_inventory(test, item_params, residual_std, "baseline_forecast")
    sim_rf   = simulate_inventory(test, item_params, residual_std, "rf_forecast")

    scm_base = compute_supply_chain_metrics(sim_base)
    scm_rf   = compute_supply_chain_metrics(sim_rf)

    outdir = PROJECT_ROOT / "outputs"
    outdir.mkdir(exist_ok=True)

    scm = pd.DataFrame({"Baseline": scm_base, "AI-RF": scm_rf})
    pd.DataFrame([fm]).round(4).to_csv(outdir / "forecast_metrics.csv",    index=False)
    scm.to_csv(outdir / "supply_chain_metrics.csv")
    feat_imp.to_csv(outdir / "feature_importance.csv", index=False)
    sim_base.to_csv(outdir / "simulation_baseline.csv", index=False)
    sim_rf.to_csv(outdir / "simulation_ai_rf.csv",     index=False)

    with open(outdir / "summary.json", "w") as f:
        json.dump({
            "forecast_metrics": fm,
            "dataset_rows": int(len(df)),
            "train_weeks": int(train["week"].nunique()),
            "test_weeks":  int(test["week"].nunique()),
            "meals": df["meal_id"].nunique(),
        }, f, indent=2, default=float)

    print("\n=== Forecast Accuracy ===")
    print(pd.DataFrame([fm]).round(2).to_string(index=False))
    print("\n=== Supply Chain KPIs ===")
    print(scm.round(2))
    print(f"\nOutputs saved to: {outdir}")
