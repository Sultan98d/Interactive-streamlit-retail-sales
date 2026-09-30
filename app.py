import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="U.S. Retail Sales Time Series Analysis",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Session State Initialization
# ---------------------------------------------------------
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

if "saved_ranges" not in st.session_state:
    st.session_state.saved_ranges = []

def increment_interactions():
    st.session_state.interaction_count += 1

# ---------------------------------------------------------
# 3. Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("RSAFSNA.csv")
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Identify date column and value column
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "RSAFSNA" if "RSAFSNA" in df.columns else df.columns[1]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col]).sort_values(date_col).reset_index(drop=True)
    
    df.rename(columns={date_col: "Date", val_col: "Sales"}, inplace=True)
    
    # Calculate YoY Growth (%)
    df["YoY_Growth"] = df["Sales"].pct_change(12) * 100
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading 'RSAFSNA.csv': {e}. Please place the dataset in the root folder.")
    st.stop()

# ---------------------------------------------------------
# 4. Sidebar Controls & Dynamic Resampling
# ---------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

# Resolution selection
resolution = st.sidebar.selectbox(
    "Time Resolution",
    options=["Monthly", "Quarterly", "Yearly"],
    index=0,
    on_change=increment_interactions
)

# Year Range Slider
min_year = int(df_raw["Date"].dt.year.min())
max_year = int(df_raw["Date"].dt.year.max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    on_change=increment_interactions
)

# Show observation points checkbox
show_points = st.sidebar.checkbox(
    "Show Observation Points",
    value=False,
    on_change=increment_interactions
)

# Bootstrap settings
st.sidebar.markdown("---")
st.sidebar.subheader("Bootstrap Settings")
n_bootstraps = st.sidebar.slider(
    "Resample Count (N)",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    on_change=increment_interactions
)

# ---------------------------------------------------------
# 5. Data Processing based on Filters
# ---------------------------------------------------------
df_filtered = df_raw[
    (df_raw["Date"].dt.year >= year_range[0]) & 
    (df_raw["Date"].dt.year <= year_range[1])
].copy()

# Resampling according to selection
if resolution == "Quarterly":
    df_resampled = df_filtered.set_index("Date").resample("QE").agg({"Sales": "mean", "YoY_Growth": "mean"}).reset_index()
elif resolution == "Yearly":
    df_resampled = df_filtered.set_index("Date").resample("YE").agg({"Sales": "mean", "YoY_Growth": "mean"}).reset_index()
else:
    df_resampled = df_filtered.copy()

# ---------------------------------------------------------
# 6. Header & Top Metrics
# ---------------------------------------------------------
st.title("📈 U.S. Retail & Food Services Sales Analysis (RSAFSNA)")
st.markdown("An interactive econometric dashboard for time-series decomposition, uncertainty modeling, and growth diagnostics.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Resolution", resolution)
m2.metric("Start Year", year_range[0])
m3.metric("End Year", year_range[1])
m4.metric("Observations", len(df_resampled))
m5.metric("Interactions", st.session_state.interaction_count)

# Save Bookmark Feature (Session State Requirement)
st.sidebar.markdown("---")
if st.sidebar.button("🔖 Bookmark Current View"):
    bookmark_label = f"{resolution} ({year_range[0]}-{year_range[1]})"
    if bookmark_label not in st.session_state.saved_ranges:
        st.session_state.saved_ranges.append(bookmark_label)

if st.session_state.saved_ranges:
    st.sidebar.subheader("Saved Bookmarks")
    for b in st.session_state.saved_ranges:
        st.sidebar.text(f"• {b}")

# ---------------------------------------------------------
# 7. Main Analysis Tabs
# ---------------------------------------------------------
tab_overview, tab_trend, tab_season, tab_bootstrap, tab_data, tab_about = st.tabs([
    "📊 Overview", 
    "📈 Trend Analysis", 
    "🍂 Seasonality", 
    "🎲 Uncertainty (Bootstrap)", 
    "📋 Filtered Data", 
    "ℹ️ About"
])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Sales Trajectory & Interactive Range Selector")
    
    # Altair Interactive Plot
    brush = alt.selection_interval(encodings=['x'])
    
    base = alt.Chart(df_resampled).mark_line(color="#1f77b4", strokeWidth=2).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Sales:Q", title="Sales (Millions of $)", scale=alt.Scale(zero=True)),
        tooltip=["Date:T", "Sales:Q", "YoY_Growth:Q"]
    )
    
    if show_points:
        points = alt.Chart(df_resampled).mark_circle(size=30, color="#d62728").encode(
            x="Date:T",
            y="Sales:Q"
        )
        chart = base + points
    else:
        chart = base

    upper = chart.encode(
        x=alt.X("Date:T", scale=alt.Scale(domain=brush))
    ).properties(width=800, height=350, title="Zoomed View (Drag on Context Chart Below)")

    lower = base.properties(
        width=800, height=100, title="Context Chart (Click & Drag to Zoom)"
    ).add_params(brush)

    st.altair_chart(upper & lower, use_container_width=True)

# --- Tab 2: Trend Analysis ---
with tab_trend:
    st.subheader("Moving Average & Trend Diagnostics")
    
    window_size = 12 if resolution == "Monthly" else (4 if resolution == "Quarterly" else 3)
    df_resampled["Rolling_Avg"] = df_resampled["Sales"].rolling(window=window_size, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_resampled["Date"], df_resampled["Sales"], label="Observed Sales", color="gray", alpha=0.5)
    ax.plot(df_resampled["Date"], df_resampled["Rolling_Avg"], label=f"{window_size}-Period Moving Average", color="navy", lw=2)
    ax.set_ylabel("Sales (Millions of $)")
    ax.set_ylim(bottom=0)
    ax.set_title("Long-Term Sales Trend")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# --- Tab 3: Seasonality ---
with tab_season:
    st.subheader("Seasonal Decomposition (Additive vs Multiplicative)")
    
    if resolution != "Monthly":
        st.warning("Seasonality analysis is best conducted on Monthly data. Switch to 'Monthly' resolution for detailed decomposition.")
    else:
        decomp_type = st.radio("Decomposition Model", ["additive", "multiplicative"], horizontal=True)
        
        df_decomp = df_filtered.set_index("Date")["Sales"].dropna()
        if len(df_decomp) >= 24:
            res = seasonal_decompose(df_decomp, model=decomp_type, period=12)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            ax1.plot(res.observed, color="black"); ax1.set_ylabel("Observed")
            ax2.plot(res.trend, color="blue"); ax2.set_ylabel("Trend")
            ax3.plot(res.seasonal, color="green"); ax3.set_ylabel("Seasonal")
            ax4.plot(res.resid, color="red"); ax4.set_ylabel("Residuals")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Insufficient data points for seasonal decomposition (Minimum 24 months required).")

# --- Tab 4: Bootstrap Uncertainty ---
with tab_bootstrap:
    st.subheader("Moving Average Confidence Interval via Bootstrap Resampling")
    
    @st.cache_data
    def run_bootstrap(data_series, n_iterations, window):
        np.random.seed(42)
        n = len(data_series)
        boot_means = []
        
        for _ in range(n_iterations):
            sample = np.random.choice(data_series, size=n, replace=True)
            rolling = pd.Series(sample).rolling(window=window, min_periods=1).mean()
            boot_means.append(rolling.values)
            
        boot_means = np.array(boot_means)
        lower_bound = np.percentile(boot_means, 2.5, axis=0)
        upper_bound = np.percentile(boot_means, 97.5, axis=0)
        return lower_bound, upper_bound

    window = 12 if resolution == "Monthly" else 4
    lower, upper = run_bootstrap(df_resampled["Sales"].values, n_bootstraps, window)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_resampled["Date"], df_resampled["Rolling_Avg"], color="navy", label="Rolling Mean")
    ax.fill_between(df_resampled["Date"], lower, upper, color="skyblue", alpha=0.4, label="95% Confidence Interval")
    ax.set_ylabel("Sales (Millions of $)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Bootstrap Uncertainty Estimation ({n_bootstraps} Iterations)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# --- Tab 5: Data ---
with tab_data:
    st.subheader("Filtered Dataset & Download")
    st.dataframe(df_resampled, use_container_width=True)
    
    csv_data = df_resampled.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Dataset",
        data=csv_data,
        file_name=f"RSAFSNA_filtered_{resolution}.csv",
        mime="text/csv"
    )

# --- Tab 6: About ---
with tab_about:
    st.markdown("""
    ### ℹ️ About this Application
    * **Data Source:** U.S. Census Bureau / FRED (Advance Retail Sales: Retail and Food Services, Total - `RSAFSNA`).
    * **Methodology:** 
        * **Resampling:** Monthly, Quarterly, and Yearly time aggregation.
        * **Decomposition:** Time series classical additive & multiplicative decomposition using `statsmodels`.
        * **Uncertainty Estimation:** Non-parametric Bootstrap resampling to estimate 95% confidence intervals around rolling statistics.
    """)
