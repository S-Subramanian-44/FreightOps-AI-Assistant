import pandas as pd
import numpy as np
from llm_utils import llm_explain

def robust_nanmean(series: pd.Series, default=np.nan):
    val = series.dropna()
    return val.mean() if len(val) else default

def fit_expected_cost_linear(df: pd.DataFrame):
    tmp = df[["Weight_kg", "Volume_m3", "Cost_USD"]].dropna()
    if len(tmp) < 5:
        return (0.0, 0.0, robust_nanmean(df["Cost_USD"], default=0.0))
    X = tmp[["Weight_kg", "Volume_m3"]].values
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([X, ones])
    y = tmp["Cost_USD"].values
    try:
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return coeffs.tolist()
    except Exception:
        return (0.0, 0.0, robust_nanmean(df["Cost_USD"], default=0.0))

def add_expected_cost(df: pd.DataFrame) -> pd.DataFrame:
    a, b, c = fit_expected_cost_linear(df)
    df = df.copy()
    w = df["Weight_kg"].fillna(0)
    v = df["Volume_m3"].fillna(0)
    df["Expected_Cost"] = a * w + b * v + c
    df["Cost_Variance"] = df["Cost_USD"] - df["Expected_Cost"]
    return df


def cost_analyzer(df: pd.DataFrame, query: str):
    work = add_expected_cost(df)
    # KPIs
    avg_cost = robust_nanmean(work["Cost_USD"])
    avg_cost_per_kg = robust_nanmean(work["Cost_per_kg"])
    avg_var = robust_nanmean(work["Cost_Variance"])

    # Top outliers by absolute variance
    outliers = work.dropna(subset=["Cost_Variance"]).copy()
    outliers["Abs_Var"] = outliers["Cost_Variance"].abs()
    outliers = outliers.sort_values("Abs_Var", ascending=False).head(10)
    show_cols = [
        "Shipment_Date", "Origin", "Destination", "Weight_kg", "Volume_m3",
        "Cost_USD", "Expected_Cost", "Cost_Variance", "Product_Type"
    ]
    outliers_table = outliers[show_cols]

    kpis = [
        {"label": "Avg Cost (USD)", "value": None if pd.isna(avg_cost) else round(float(avg_cost), 2)},
        {"label": "Avg Cost per kg (USD/kg)", "value": None if pd.isna(avg_cost_per_kg) else round(float(avg_cost_per_kg), 4)},
        {"label": "Avg Variance (USD)", "value": None if pd.isna(avg_var) else round(float(avg_var), 2)},
        {"label": "Outliers (top 10)", "value": len(outliers_table)},
    ]

    context = (
        "Cost Analysis KPIs:\n"
        f"- Average Cost: {kpis[0]['value']}\n"
        f"- Average Cost per kg: {kpis[1]['value']}\n"
        f"- Average Variance: {kpis[2]['value']}\n"
        f"- Outliers Listed: {kpis[3]['value']}\n\n"
        "Outlier Samples (top rows):\n" + outliers_table.head(5).to_csv(index=False)
    )
    explanation = llm_explain("a freight cost analyst", context, query)
    return explanation, kpis, outliers_table


def route_planner(df: pd.DataFrame, query: str):
    # Rank routes by affordability and stability (frequency, low variance)
    work = add_expected_cost(df)
    route_stats = (
        work.groupby(["Origin", "Destination"])
        .agg(
            count=("Cost_USD", "count"),
            avg_cost=("Cost_USD", "mean"),
            avg_cost_per_kg=("Cost_per_kg", "mean"),
            var_cost=("Cost_USD", "var"),
        )
        .reset_index()
    )

    if len(route_stats) == 0:
        kpis = [{"label": "Routes Found", "value": 0}]
        return "No routes found.", kpis, route_stats

    # Normalize for scoring
    rs = route_stats.copy()
    # Avoid division by zero by replacing NaNs
    rs["avg_cost"] = rs["avg_cost"].fillna(rs["avg_cost"].mean())
    rs["var_cost"] = rs["var_cost"].fillna(rs["var_cost"].mean())
    rs["avg_cost_per_kg"] = rs["avg_cost_per_kg"].fillna(rs["avg_cost_per_kg"].mean())

    # Min-max normalize
    def minmax(s: pd.Series):
        if s.nunique() <= 1:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    rs["cost_n"] = minmax(rs["avg_cost"])
    rs["var_n"] = minmax(rs["var_cost"])
    rs["freq_n"] = minmax(rs["count"])

    # Lower cost and lower variance are better; higher frequency is better
    rs["score"] = (0.6 * (1 - rs["cost_n"])) + (0.2 * (1 - rs["var_n"])) + (0.2 * rs["freq_n"])

    ranking = rs.sort_values("score", ascending=False).head(10)
    kpis = [
        {"label": "Total Routes", "value": int(route_stats.shape[0])},
        {"label": "Top Routes Shown", "value": int(ranking.shape[0])},
        {"label": "Best Score", "value": round(float(ranking["score"].iloc[0]), 3) if len(ranking) else None},
    ]

    context = (
        "Route Planning Insights:\n"
        "We ranked routes by low cost, low variance, and higher frequency.\n\n"
        "Top 5 routes (Origin, Destination, score, avg_cost, count):\n" +
        ranking[["Origin", "Destination", "score", "avg_cost", "count"]].head(5).to_csv(index=False)
    )
    explanation = llm_explain("a route planner focusing on cost and reliability", context, query)
    return explanation, kpis, ranking[["Origin", "Destination", "count", "avg_cost", "avg_cost_per_kg", "var_cost", "score"]]


def compliance_checker(df: pd.DataFrame, query: str):
    work = df.copy()
    flags = []

    # Simple heuristic rules
    hazmat_like = {"Ethanol", "Acetone", "Diesel", "Gasoline", "Paint", "Batteries"}
    high_cost_threshold = work["Cost_USD"].quantile(0.95) if work["Cost_USD"].notna().any() else np.nan
    high_weight_threshold = work["Weight_kg"].quantile(0.95) if work["Weight_kg"].notna().any() else np.nan
    high_volume_threshold = work["Volume_m3"].quantile(0.95) if work["Volume_m3"].notna().any() else np.nan

    def add_flag(row, reason):
        flags.append({
            "Shipment_Date": row.get("Shipment_Date"),
            "Origin": row.get("Origin"),
            "Destination": row.get("Destination"),
            "Product_Type": row.get("Product_Type"),
            "Cost_USD": row.get("Cost_USD"),
            "Weight_kg": row.get("Weight_kg"),
            "Volume_m3": row.get("Volume_m3"),
            "Reason": reason
        })

    for _, r in work.iterrows():
        # Missing critical fields
        if pd.isna(r["Shipment_Date"]) or pd.isna(r["Origin"]) or pd.isna(r["Destination"]) or pd.isna(r["Cost_USD"]):
            add_flag(r, "Missing critical fields")

        # Hazmat-like product types
        if isinstance(r["Product_Type"], str) and r["Product_Type"].strip() in hazmat_like:
            add_flag(r, "Potential hazardous cargo; ensure hazmat documentation")

        # High-risk thresholds
        if pd.notna(high_cost_threshold) and pd.notna(r["Cost_USD"]) and r["Cost_USD"] >= high_cost_threshold:
            add_flag(r, "High cost shipment; additional review recommended")

        if pd.notna(high_weight_threshold) and pd.notna(r["Weight_kg"]) and r["Weight_kg"] >= high_weight_threshold:
            add_flag(r, "Very heavy shipment; verify equipment and permits")

        if pd.notna(high_volume_threshold) and pd.notna(r["Volume_m3"]) and r["Volume_m3"] >= high_volume_threshold:
            add_flag(r, "Very large volume; verify dimensional/oversize paperwork")

        # Weekend shipments: may require special handling
        if pd.notna(r["Shipment_Date"]):
            weekday = int(pd.to_datetime(r["Shipment_Date"]).weekday())  # 5=Sat, 6=Sun
            if weekday >= 5:
                add_flag(r, "Weekend shipment; confirm carrier/terminal availability")

    flags_df = pd.DataFrame(flags)
    kpis = [
        {"label": "Total Shipments", "value": int(df.shape[0])},
        {"label": "Flags Found", "value": int(flags_df.shape[0])},
        {"label": "Unique Routes Flagged", "value": int(flags_df[["Origin", "Destination"]].dropna().drop_duplicates().shape[0]) if not flags_df.empty else 0},
    ]

    sample_flags_csv = flags_df.head(10).to_csv(index=False) if not flags_df.empty else "No flags."
    context = (
        "Compliance Check Summary:\n"
        f"- Total Flags: {kpis[1]['value']}\n\n"
        "Sample Flags (top rows):\n" + sample_flags_csv
    )
    explanation = llm_explain("a logistics compliance officer", context, query)
    return explanation, kpis, flags_df.head(50)


def summary_agent(df: pd.DataFrame, query: str):
    # Monthly and weekly KPIs
    monthly = (
        df.groupby("Month")
        .agg(
            shipments=("Cost_USD", "count"),
            total_cost=("Cost_USD", "sum"),
            avg_cost=("Cost_USD", "mean"),
            avg_cost_per_kg=("Cost_per_kg", "mean"),
        )
        .reset_index()
        .sort_values("Month", ascending=True)
    )

    weekly = (
        df.groupby("Week")
        .agg(
            shipments=("Cost_USD", "count"),
            total_cost=("Cost_USD", "sum"),
            avg_cost=("Cost_USD", "mean"),
            avg_cost_per_kg=("Cost_per_kg", "mean"),
        )
        .reset_index()
        .sort_values("Week", ascending=True)
    )

    overall = {
        "Total Shipments": int(df.shape[0]),
        "Avg Cost (USD)": None if pd.isna(df["Cost_USD"].mean()) else round(float(df["Cost_USD"].mean()), 2),
        "Avg Cost per kg (USD/kg)": None if pd.isna(df["Cost_per_kg"].mean()) else round(float(df["Cost_per_kg"].mean()), 4),
        "Unique Routes": int(df[["Origin", "Destination"]].dropna().drop_duplicates().shape[0]),
    }
    kpis = [{"label": k, "value": v} for k, v in overall.items()]

    context = (
        "Summary KPIs:\n" + "\n".join([f"- {k}: {v}" for k, v in overall.items()]) +
        "\n\nMonthly Snapshot (top rows):\n" + monthly.head(6).to_csv(index=False) +
        "\n\nWeekly Snapshot (top rows):\n" + weekly.head(6).to_csv(index=False)
    )
    explanation = llm_explain("a logistics analyst summarizing performance", context, query)
    return explanation, kpis, {"monthly": monthly, "weekly": weekly}


# --------------------------
# Orchestrator
# --------------------------
def orchestrator(query: str):
    q = (query or "").lower()
    if any(k in q for k in ["cost", "variance", "over budget", "budget"]):
        return "Cost Analyzer", cost_analyzer
    if any(k in q for k in ["route", "carrier", "transit", "best path"]):
        return "Route Planner", route_planner
    if any(k in q for k in ["compliance", "customs", "flag", "documentation", "permit"]):
        return "Compliance Checker", compliance_checker
    return "Summary Agent", summary_agent
