# AI-Based Demand Forecasting & Inventory Optimization for Food Delivery Supply Chains

**INSE 6290 – Quality in Supply Chain Design · Concordia University · Winter 2026**

Team: Md Golam Raiyhan · Dylan Zuzarte · Afsana Tasnim Nishat · Shadid Alam Al Huda

---

## What This Project Does

This prototype implements a **5-layer AI-powered supply chain system** for perishable food delivery,
using **real demand data** (456,548 rows, 145 weeks, 77 fulfillment centers, 51 meals).

1. **Data** – Real foodDemand dataset, top 10 meals aggregated weekly across all 77 centers
2. **Feature Engineering** – Lag-1, Lag-4, Rolling-4, Rolling-8, promotions, price discount ratio
3. **Forecasting** – Baseline (Lag-4) vs. AI (Random Forest, 300 trees)
4. **Inventory** – Newsvendor ordering + FEFO aging model (weekly periods)
5. **Evaluation** – Fill Rate, Waste %, Inventory Turnover, Gross Profit, Stockout Rate

### Key Results (10-week real-data test horizon)
| Metric | Baseline | AI-RF | Change |
|--------|----------|-------|--------|
| MAE | 11,698 | 4,386 | **−62.5%** |
| Food Waste % | 48% | 11% | **−77%** |
| Inventory Turnover | 24× | 84× | **+250%** |
| Fill Rate | 100% | ~100% | — |
| Gross Profit | −$807M | +$342M | **+$1.15B** |

---

## Project Structure

```
INSE 6290 project/
├── app.py                        ← Streamlit interactive demo  ← START HERE
├── run_project.py                ← ML pipeline (forecast + simulate)
├── generate_synthetic_data.py    ← Synthetic dataset generator
├── synthetic_food_orders.csv     ← Generated dataset (900 rows)
├── requirements.txt              ← Python dependencies
├── outputs/                      ← Generated plots, CSVs, JSON
│   ├── forecast_metrics.csv
│   ├── supply_chain_metrics.csv
│   ├── feature_importance.csv
│   ├── simulation_baseline.csv
│   ├── simulation_ai_rf.csv
│   ├── summary.json
│   └── *.png
├── foodDemand_train/             ← External historical dataset (reference)
├── food_Demand_test.csv          ← External test data (reference)
├── INSE_6290_Full_Project_Report.docx
└── New_Presentation_6290.pptx
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the interactive demo (recommended)
```bash
streamlit run app.py
```
Opens at http://localhost:8501 in your browser.

### 3. Or run the pipeline in the terminal
```bash
python3 generate_synthetic_data.py   # regenerate dataset
python3 run_project.py               # train + simulate + save outputs
```

---

## Demo Pages (Streamlit App)

| Page | What You'll See |
|------|----------------|
| 🏠 Project Overview | Architecture, heatmap, daily demand chart |
| 📈 Demand Forecasting | Interactive per-item forecast comparison |
| 📦 Inventory Simulation | Per-item daily simulation (order, waste, inventory) |
| 🏆 KPI Dashboard | Side-by-side supply chain metrics |
| 🔍 Feature Importance | What drives the Random Forest predictions |
| 📋 Raw Data | Browse all datasets interactively |

---

## Technology Stack
- Python 3.9+
- Pandas / NumPy – data manipulation
- scikit-learn – Random Forest forecasting
- Matplotlib – static plots
- Streamlit + Plotly – interactive web demo
