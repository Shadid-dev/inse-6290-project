"""
INSE 6290 – AI-Based Demand Forecasting & Inventory Optimization
Concordia University, Winter 2026

Real Dataset: foodDemand_train (456k rows, 145 weeks, 77 centers, 51 meals)
Run:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    load_real_data,
    train_and_forecast,
    simulate_inventory,
    compute_supply_chain_metrics,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="INSE 6290 – AI Demand Forecasting",
    page_icon="🍔",
    layout="wide",
)

# ──────────────────────────────────────────────
# Run pipeline once per session (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Running ML pipeline on real data — please wait…")
def run_pipeline():
    df = load_real_data(top_n_meals=10)
    train, test, fm, residual_std, item_params, feat_imp = train_and_forecast(df)
    sim_base = simulate_inventory(test, item_params, residual_std, "baseline_forecast")
    sim_rf   = simulate_inventory(test, item_params, residual_std, "rf_forecast")
    scm_base = compute_supply_chain_metrics(sim_base)
    scm_rf   = compute_supply_chain_metrics(sim_rf)
    return df, train, test, fm, feat_imp, sim_base, sim_rf, scm_base, scm_rf


df, train, test, fm, feat_imp, sim_base, sim_rf, scm_base, scm_rf = run_pipeline()

# Meal labels for dropdowns
meal_labels = (
    df.groupby("meal_id")[["category", "cuisine"]]
    .first()
    .reset_index()
    .assign(label=lambda x: x["meal_id"].astype(str) + " – " + x["category"] + " (" + x["cuisine"] + ")")
)
meal_map   = dict(zip(meal_labels["label"], meal_labels["meal_id"]))
meal_names = sorted(meal_labels["label"].tolist())

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍔 INSE 6290")
    st.markdown("**AI Supply Chain Demo**")
    st.markdown("Concordia University · Winter 2026")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Project Overview",
            "📈 Demand Forecasting",
            "📦 Inventory Simulation",
            "🏆 KPI Dashboard",
            "🔍 Feature Importance",
            "📋 Data Explorer",
        ],
    )
    st.divider()
    st.caption("Dataset: foodDemand_train")
    st.caption("456,548 rows · 145 weeks · 77 centers · 51 meals")
    st.caption("Team: Raiyhan · Dylan · Tasnim · Shadid")


# ══════════════════════════════════════════════
# PAGE 1 – Project Overview
# ══════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.title("AI-Based Demand Forecasting & Inventory Optimization")
    st.subheader("INSE 6290 · Quality in Supply Chain Design · Concordia University · Winter 2026")
    st.divider()

    left, right = st.columns([3, 1])
    with left:
        st.markdown("""
### Problem
Food delivery companies face two competing risks:
- **Over-ordering** → spoilage, holding costs, financial loss
- **Under-ordering** → stockouts, lost revenue, customer dissatisfaction

Traditional methods (moving averages) fail to capture promotional spikes, price elasticity,
and multi-week demand cycles.

### Dataset Used
| Property | Value |
|----------|-------|
| Source | Real foodDemand dataset (Kaggle / industry) |
| Rows | **456,548** transactions |
| Time span | **145 weeks (~2.8 years)** |
| Fulfillment centers | **77** |
| Unique meals | **51** |
| Focus | Top 10 meals by volume, aggregated across all centers |
| Extra signals | Email promotions, homepage features, price discounts |

### Solution: 5-Layer AI Supply Chain
| Layer | Component |
|-------|-----------|
| **Data** | Real demand + augmented cost params from actual prices |
| **Features** | Lag-1, Lag-4, Rolling-4, Rolling-8, promotions, price ratio |
| **Forecasting** | Baseline (Lag-4) vs. Random Forest (300 trees) |
| **Inventory** | Newsvendor ordering + FEFO aging model (weekly periods) |
| **Evaluation** | Fill Rate, Waste %, Turnover, Profit, Stockout Rate |
        """)

    with right:
        st.metric("Meals Analyzed", 10)
        st.metric("Training Weeks", int(train["week"].nunique()))
        st.metric("Test Horizon", "10 weeks")
        mae_imp = (fm["Baseline_MAE"] - fm["RF_MAE"]) / fm["Baseline_MAE"] * 100
        st.metric("MAE Improvement", f"{mae_imp:.1f}%", help="AI vs Baseline")
        waste_imp = (scm_base["Food Waste %"] - scm_rf["Food Waste %"]) / scm_base["Food Waste %"] * 100
        st.metric("Waste Reduction", f"{waste_imp:.0f}%", help="AI vs Baseline")
        profit_diff = scm_rf["Gross Profit ($)"] - scm_base["Gross Profit ($)"]
        st.metric("Extra Profit (AI)", f"${profit_diff/1e6:.0f}M", help="AI vs Baseline over test period")

    st.divider()

    # Aggregate weekly demand trend
    weekly_all = df.groupby("week")["num_orders"].sum().reset_index()
    fig = px.area(weekly_all, x="week", y="num_orders",
                  title="Total Weekly Demand – Top 10 Meals Across All 77 Centers",
                  labels={"num_orders": "Weekly Orders", "week": "Week"},
                  color_discrete_sequence=["#1f77b4"])
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    # Promotion impact
    st.subheader("Real Promotional Impact")
    promo_comp = df.groupby("emailer_for_promotion")["num_orders"].mean().reset_index()
    promo_comp["emailer_for_promotion"] = promo_comp["emailer_for_promotion"].map({0: "No Promo", 1: "Email Promo"})
    fig2 = px.bar(promo_comp, x="emailer_for_promotion", y="num_orders",
                  color="emailer_for_promotion",
                  color_discrete_map={"No Promo": "#aec7e8", "Email Promo": "#1f77b4"},
                  title="Average Weekly Orders: With vs Without Email Promotion",
                  labels={"num_orders": "Avg Weekly Orders", "emailer_for_promotion": ""},
                  text_auto=True)
    fig2.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 – Demand Forecasting
# ══════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting")
    st.caption("Baseline (Lag-4 weeks) vs. AI Random Forest — 10-week test horizon")

    selected_label = st.selectbox("Select Meal", meal_names)
    selected_meal  = meal_map[selected_label]
    st.divider()

    sub = test[test["meal_id"] == selected_meal].sort_values("week")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["week"], y=sub["num_orders"],
                             mode="lines+markers", name="Actual Demand",
                             line=dict(color="#333", width=2.5), marker_size=8))
    fig.add_trace(go.Scatter(x=sub["week"], y=sub["baseline_forecast"],
                             mode="lines+markers", name="Baseline (Lag-4)",
                             line=dict(color="#d62728", dash="dash"), marker_size=6))
    fig.add_trace(go.Scatter(x=sub["week"], y=sub["rf_forecast"],
                             mode="lines+markers", name="AI – Random Forest",
                             line=dict(color="#2ca02c"), marker_size=7, marker_symbol="diamond"))
    # Highlight promo weeks
    promo_weeks = sub[sub["emailer_for_promotion"] == 1]["week"]
    for pw in promo_weeks:
        fig.add_vline(x=pw, line_dash="dot", line_color="orange", opacity=0.6,
                      annotation_text="📧 Promo", annotation_position="top")
    fig.update_layout(
        title=f"{selected_label}: Actual vs Forecast (Test Weeks)",
        xaxis_title="Week", yaxis_title="Weekly Orders",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy metrics
    st.subheader("Forecast Accuracy — All 10 Meals Combined")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline MAE",  f"{fm['Baseline_MAE']:,.0f}")
    c2.metric("AI-RF MAE",     f"{fm['RF_MAE']:,.0f}",
              delta=f"{fm['RF_MAE'] - fm['Baseline_MAE']:+,.0f}", delta_color="inverse")
    c3.metric("Baseline RMSE", f"{fm['Baseline_RMSE']:,.0f}")
    c4.metric("AI-RF RMSE",    f"{fm['RF_RMSE']:,.0f}",
              delta=f"{fm['RF_RMSE'] - fm['Baseline_RMSE']:+,.0f}", delta_color="inverse")

    st.divider()
    st.subheader("Per-Meal MAE Comparison")
    rows = []
    for mid in test["meal_id"].unique():
        s = test[test["meal_id"] == mid]
        lbl = meal_labels[meal_labels["meal_id"] == mid]["label"].values[0]
        rows.append({
            "Meal": lbl,
            "Baseline MAE": int(np.mean(np.abs(s["num_orders"] - s["baseline_forecast"]))),
            "AI-RF MAE":    int(np.mean(np.abs(s["num_orders"] - s["rf_forecast"]))),
        })
    pim = pd.DataFrame(rows).sort_values("Baseline MAE", ascending=False)
    pim["Improvement %"] = ((pim["Baseline MAE"] - pim["AI-RF MAE"]) / pim["Baseline MAE"] * 100).round(1)

    fig2 = go.Figure()
    fig2.add_bar(x=pim["Meal"], y=pim["Baseline MAE"], name="Baseline", marker_color="#d62728")
    fig2.add_bar(x=pim["Meal"], y=pim["AI-RF MAE"],    name="AI-RF",   marker_color="#2ca02c")
    fig2.update_layout(barmode="group", title="MAE by Meal",
                       xaxis_title="Meal", yaxis_title="MAE (weekly orders)",
                       xaxis_tickangle=-30, height=400)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pim, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 3 – Inventory Simulation
# ══════════════════════════════════════════════
elif page == "📦 Inventory Simulation":
    st.title("📦 Inventory Simulation")
    st.caption("Newsvendor ordering policy + FEFO (First-Expire-First-Out) shelf-life model — weekly periods")

    policy   = st.radio("Policy", ["Baseline (Lag-4)", "AI – Random Forest"], horizontal=True)
    sim      = sim_base if "Baseline" in policy else sim_rf
    sel_lbl  = st.selectbox("Select Meal", meal_names)
    sel_meal = meal_map[sel_lbl]
    st.divider()

    s = sim[sim["meal_id"] == sel_meal].sort_values("week")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["week"], y=s["order_qty"],
                         name="Order Qty", marker_color="#aec7e8", opacity=0.55))
    fig.add_trace(go.Scatter(x=s["week"], y=s["actual_demand"],
                             mode="lines+markers", name="Actual Demand",
                             line=dict(color="#333", width=2.5)))
    fig.add_trace(go.Scatter(x=s["week"], y=s["forecast"],
                             mode="lines+markers", name="Forecast",
                             line=dict(color="#1f77b4", dash="dot")))
    fig.add_trace(go.Scatter(x=s["week"], y=s["waste_units"],
                             mode="lines+markers", name="Waste",
                             line=dict(color="#d62728"), marker_size=9))
    fig.add_trace(go.Scatter(x=s["week"], y=s["ending_inventory"],
                             mode="lines+markers", name="End Inventory",
                             line=dict(color="#9467bd")))
    # Promo weeks
    promo_wks = s[s["emailer_for_promotion"] == 1]["week"]
    for pw in promo_wks:
        fig.add_vline(x=pw, line_dash="dot", line_color="orange", opacity=0.6)
    fig.update_layout(
        title=f"{sel_lbl} – Weekly Inventory ({policy})",
        xaxis_title="Week", yaxis_title="Units",
        barmode="overlay", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aggregated Weekly Outcomes — All 10 Meals")
    agg = sim.groupby("week")[["sales", "stockout_units", "waste_units"]].sum().reset_index()
    fig2 = px.line(agg, x="week", y=["sales", "stockout_units", "waste_units"],
                   color_discrete_map={"sales": "#2ca02c", "stockout_units": "#d62728",
                                       "waste_units": "#ff7f0e"},
                   title=f"Aggregated Weekly Outcomes – {policy}",
                   labels={"value": "Units", "week": "Week", "variable": "Metric"})
    fig2.update_layout(height=360)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detail Table")
    st.dataframe(
        s[["week","forecast","actual_demand","order_qty","sales",
           "stockout_units","waste_units","ending_inventory",
           "emailer_for_promotion","homepage_featured"]].round(0).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════
# PAGE 4 – KPI Dashboard
# ══════════════════════════════════════════════
elif page == "🏆 KPI Dashboard":
    st.title("🏆 Supply Chain KPI Dashboard")
    st.caption("Baseline (Lag-4) vs. AI Random Forest — 10-week real-data simulation")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fill Rate (AI-RF)",
              f"{scm_rf['Fill Rate']*100:.2f}%",
              delta=f"{(scm_rf['Fill Rate'] - scm_base['Fill Rate'])*100:+.2f}pp")
    c2.metric("Food Waste % (AI-RF)",
              f"{scm_rf['Food Waste %']*100:.2f}%",
              delta=f"{(scm_rf['Food Waste %'] - scm_base['Food Waste %'])*100:+.2f}pp",
              delta_color="inverse")
    c3.metric("Inv. Turnover (AI-RF)",
              f"{scm_rf['Inventory Turnover']:.1f}×",
              delta=f"{scm_rf['Inventory Turnover'] - scm_base['Inventory Turnover']:+.1f}×")
    profit_diff = scm_rf["Gross Profit ($)"] - scm_base["Gross Profit ($)"]
    c4.metric("Profit Gain (AI-RF)",
              f"${profit_diff/1e6:.0f}M",
              delta="vs Baseline policy")

    st.divider()

    # Rate KPIs bar
    pct_kpis = ["Fill Rate", "Stockout Rate", "Food Waste %"]
    fig = go.Figure()
    fig.add_bar(x=pct_kpis,
                y=[scm_base[k]*100 for k in pct_kpis],
                name="Baseline", marker_color="#d62728",
                text=[f"{scm_base[k]*100:.2f}%" for k in pct_kpis],
                textposition="outside")
    fig.add_bar(x=pct_kpis,
                y=[scm_rf[k]*100 for k in pct_kpis],
                name="AI-RF", marker_color="#2ca02c",
                text=[f"{scm_rf[k]*100:.2f}%" for k in pct_kpis],
                textposition="outside")
    fig.update_layout(barmode="group", title="Rate KPIs (%)",
                      yaxis_title="%", height=380)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure(go.Bar(
            x=["Baseline", "AI-RF"],
            y=[scm_base["Total Ordered"]/1e6, scm_rf["Total Ordered"]/1e6],
            marker_color=["#d62728", "#2ca02c"],
            text=[f"{scm_base['Total Ordered']/1e6:.2f}M", f"{scm_rf['Total Ordered']/1e6:.2f}M"],
            textposition="outside"
        ))
        fig2.update_layout(title="Total Units Ordered (10-week)", yaxis_title="Million units", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        profits = [scm_base["Gross Profit ($)"]/1e6, scm_rf["Gross Profit ($)"]/1e6]
        fig3 = go.Figure(go.Bar(
            x=["Baseline", "AI-RF"],
            y=profits,
            marker_color=["#d62728", "#2ca02c"],
            text=[f"${p:.0f}M" for p in profits],
            textposition="outside"
        ))
        fig3.update_layout(title="Gross Profit (Revenue − COGS − Waste Cost)",
                           yaxis_title="$ Million", height=340)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Full Financial & KPI Table")
    kpi_df = pd.DataFrame({"Baseline": scm_base, "AI-RF": scm_rf})
    kpi_df["Change (AI−Baseline)"] = kpi_df["AI-RF"] - kpi_df["Baseline"]
    st.dataframe(kpi_df.round(2), use_container_width=True)

    st.success(
        f"**Key Takeaway:** The AI-RF policy orders **{(scm_base['Total Ordered'] - scm_rf['Total Ordered'])/1e6:.1f}M fewer units**, "
        f"cuts food waste from **{scm_base['Food Waste %']*100:.0f}%** to **{scm_rf['Food Waste %']*100:.0f}%**, "
        f"and generates an additional **${profit_diff/1e6:.0f}M in profit** over the 10-week test period."
    )


# ══════════════════════════════════════════════
# PAGE 5 – Feature Importance
# ══════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance – Random Forest")
    st.caption("Which signals drive AI demand forecasts on real data?")
    st.divider()

    fs = feat_imp.sort_values("Importance", ascending=True)
    colors = px.colors.sequential.Blues_r
    bar_colors = [colors[i % len(colors)] for i in range(len(fs))]

    fig = go.Figure(go.Bar(
        x=fs["Importance"], y=fs["Feature"],
        orientation="h",
        marker_color=bar_colors,
        text=fs["Importance"].apply(lambda v: f"{v:.1%}"),
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Importance – Random Forest (300 estimators, real data)",
        xaxis_title="Importance Score", yaxis_title="Feature",
        xaxis=dict(range=[0, fs["Importance"].max() * 1.25]),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Descriptions")
    st.markdown("""
| Feature | Description |
|---------|-------------|
| **lag_1** | Demand from 1 week ago |
| **lag_4** | Demand from 4 weeks ago (~1 month) |
| **rolling_4** | 4-week rolling average (short-term trend) |
| **rolling_8** | 8-week rolling average (medium-term trend) |
| **week_of_year** | Seasonal week index (1–52 cycle) |
| **emailer_for_promotion** | Email campaign active this week (real signal — 175% demand boost) |
| **homepage_featured** | Meal featured on homepage (real signal) |
| **price_ratio** | checkout_price / base_price (discount indicator) |
| **price_discount** | base_price − checkout_price (absolute discount in $) |
| **meal_id** | Meal identifier (item-level base demand) |

**Key Insight:** On real data, the AI model learns from *actual* promotional and pricing signals
— this is why it achieves **62% lower MAE** vs. the baseline, compared to only 25% on synthetic data.
    """)
    st.dataframe(feat_imp.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 6 – Data Explorer
# ══════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Raw Real Data", "Test Forecasts", "Simulation Results", "Dataset Summary"
    ])

    with tab1:
        st.caption(f"{df.shape[0]} rows · {df['meal_id'].nunique()} meals · weeks {df.week.min()}–{df.week.max()}")
        meals_sel = st.multiselect("Filter by meal", meal_names, default=meal_names[:3])
        ids_sel   = [meal_map[m] for m in meals_sel]
        st.dataframe(df[df["meal_id"].isin(ids_sel)].reset_index(drop=True),
                     use_container_width=True, height=420)

    with tab2:
        st.caption(f"{test.shape[0]} rows, weeks {test.week.min()}–{test.week.max()}")
        cols = ["week", "meal_id", "category", "cuisine", "num_orders",
                "baseline_forecast", "rf_forecast",
                "emailer_for_promotion", "homepage_featured", "price_ratio"]
        st.dataframe(test[cols].round(1).reset_index(drop=True),
                     use_container_width=True, height=420)

    with tab3:
        p = st.radio("Policy", ["Baseline", "AI-RF"], horizontal=True)
        show = sim_base if p == "Baseline" else sim_rf
        st.caption(f"{show.shape[0]} rows")
        st.dataframe(show.round(1).reset_index(drop=True),
                     use_container_width=True, height=420)

    with tab4:
        st.markdown("### Real Dataset Summary")
        raw = pd.read_csv(Path(__file__).resolve().parent / "foodDemand_train" / "train.csv")
        ml  = pd.read_csv(Path(__file__).resolve().parent / "foodDemand_train" / "meal_info.csv")
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Total rows (train.csv) | {len(raw):,} |
| Weeks | {raw.week.min()} – {raw.week.max()} |
| Unique meal IDs | {raw.meal_id.nunique()} |
| Unique centers | {raw.center_id.nunique()} |
| Meal categories | {', '.join(ml.category.unique()[:6])}… |
| Cuisines | {', '.join(ml.cuisine.unique())} |
| Min weekly orders | {int(raw.num_orders.min()):,} |
| Max weekly orders | {int(raw.num_orders.max()):,} |
| Email promo rate | {raw.emailer_for_promotion.mean():.1%} |
| Homepage featured rate | {raw.homepage_featured.mean():.1%} |
        """)
        st.markdown("### Cost Parameters (Derived from Actual Prices)")
        st.dataframe(
            df.groupby(["meal_id","category","cuisine"]).agg(
                avg_selling_price=("selling_price","mean"),
                avg_unit_cost=("unit_cost","mean"),
                holding_cost=("holding_cost","mean"),
                shortage_cost=("shortage_cost","mean"),
                shelf_life_weeks=("shelf_life_weeks","first"),
            ).round(2).reset_index(),
            use_container_width=True, hide_index=True
        )
