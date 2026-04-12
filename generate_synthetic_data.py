
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_food_data(output_path: str, n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    items = [
        ("Burger",     42, 1.20, 0),
        ("Pizza",      36, 1.10, 1),
        ("Pasta",      28, 0.95, 2),
        ("Salad",      24, 0.85, 3),
        ("Wrap",       26, 0.90, 4),
        ("Fried Rice", 31, 1.00, 5),
        ("Noodles",    29, 0.98, 6),
        ("Sandwich",   22, 0.80, 7),
        ("Taco",       27, 0.92, 8),
        ("Soup",       18, 0.70, 9),
    ]

    rows = []
    weekly_pattern = np.array([0.92, 0.95, 1.00, 1.03, 1.12, 1.25, 1.15])  # Mon-Sun

    for item_name, base_demand, price_factor, item_id in items:
        level_shift = rng.normal(0, 1.0)
        for i, date in enumerate(dates):
            dow = date.dayofweek
            trend       = 1.0 + 0.0018 * i
            weekend_boost = 1.08 if dow in [4, 5, 6] else 1.0
            promo       = 1.18 if (i % 21 in [6, 7]) else 1.0
            noise       = rng.normal(0, 3.5 + 0.05 * base_demand)
            demand      = (base_demand + level_shift) * weekly_pattern[dow] * trend * weekend_boost * promo + noise
            demand      = max(0, int(round(demand)))
            rows.append({
                "date":           date,
                "item_id":        item_id,
                "item_name":      item_name,
                "day_of_week":    dow,
                "is_weekend":     int(dow >= 5),
                "demand":         demand,
                "unit_cost":      round(4.0 + price_factor, 2),
                "selling_price":  round(8.0 + 2.5 * price_factor, 2),
                "holding_cost":   round(0.35 + 0.03 * price_factor, 2),
                "shortage_cost":  round(1.80 + 0.10 * price_factor, 2),
                "shelf_life_days": 3 if item_name in ["Salad", "Soup", "Sandwich"] else 4,
                "lead_time_days": 1
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    # Save to project root (same folder as this script)
    out = Path(__file__).resolve().parent / "synthetic_food_orders.csv"
    df = generate_synthetic_food_data(str(out))
    print(f"Saved {len(df)} rows to {out}")
