import os
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(ttl=600)
def load_data(relative_path) -> pd.DataFrame:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, relative_path)
    df = pd.read_csv(CSV_PATH)
    
    # Normalize columns
    df["Weight_kg"] = pd.to_numeric(df["Weight_kg"], errors="coerce")
    df["Volume_m3"] = pd.to_numeric(df["Volume_m3"], errors="coerce")
    df["Cost_USD"] = pd.to_numeric(df["Cost_USD"], errors="coerce")
    df["Shipment_Date"] = pd.to_datetime(df["Shipment_Date"], format="%d-%b-%y", errors="coerce")

    # Derived features
    safe_weight = df["Weight_kg"].replace({0: np.nan})
    df["Cost_per_kg"] = df["Cost_USD"] / safe_weight
    df["Month"] = df["Shipment_Date"].dt.to_period("M").astype(str)
    df["Week"] = df["Shipment_Date"].dt.strftime("%G-W%V")
    return df
