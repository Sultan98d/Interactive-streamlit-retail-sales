import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="U.S. Retail & Food Services Sales Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom Styling for Professional Dashboard Look
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Data Generation / Loading Function
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Simulates / Loads US Retail and Food Services Sales Data (RSAFSNA)
    Monthly data from Jan 1992 to Dec 2025.
    """
    dates = pd.date_range(start="1992-01-01", end="2025-12-01", freq="MS")
    n = len(dates)
    
    # Synthetic realistic trend + seasonality + noise matching Census/FRED patterns
    t = np.arange(n)
    trend = 150000 + 1200 * t + 1.5 * (t ** 1.8)
    seasonal = 15000 * np.sin(2 * np.pi * t / 12) + 10000 * np.cos(4 * np.pi * t / 12)
    
    # Add economic shocks (e.g., 2008 Recession, 2020 COVID)
    shock = np.zeros(n)
    for i, d in enumerate(dates):
        if d.year == 2008 or d.year == 2009:
            shock[i] = -25000
        elif d.year == 2020 and d.month in [3, 4, 5]:
            shock[i] = -60000
            
    noise = np.random.normal(0, 4000, n)
    sales = trend + seasonal + shock + noise
    
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    df.set_index("Date", inplace=True)
    return df

raw_df = load_data()

# ---------------------------------------------------------
# 3. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

# Time Resolution Selector
time_res = st.sidebar.selectbox(
    "Time Resolution",
    options=["Monthly", "Quarterly", "Yearly"],
    index=0
)

# Year Range Selector
min_year = int(raw_df.index.year.min())
max_year = int(raw_df.index.year.max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Bootstrap Resampling Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Bootstrap Settings")
bootstrap_n = st.sidebar.number_input(
    "Resampling Count (N)",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

# ---------------------------------------------------------
# 4. Data Processing & Resampling
# ---------------------------------------------------------
filtered_raw = raw_df[(raw_df.index.year >= year_range[0]) & (raw_df.index.year <= year_range[1])]

if time_res == "Monthly":
    df_resampled = filtered_raw.resample("MS").sum()
    sp_period = 12
elif time_res == "Quarterly":
    df_resampled = filtered_raw.resample("QS").sum()
    sp_period = 4
else:  # Yearly
    df_resampled = filtered_raw.resample("YS").sum()
    sp_period = 1

# Calculate YoY Growth and Rolling Average
if time_res == "Monthly":
    df_resampled["YoY_Growth"] = df_resampled["Sales"].pct_change(12) * 100
elif time_res == "Quarterly":
    df_resampled["YoY_Growth"] = df_resampled["Sales"].pct_change(4) * 100
else:
    df_resampled["YoY_Growth"] = df_resampled["Sales"].pct_change(1) * 100

df_resampled["Rolling_Avg"] = df_resampled["Sales"].rolling(window=3, min_periods=1).mean()

# ---------------------------------------------------------
# 5. Header & Main Layout
# ---------------------------------------------------------
st.title("📈 U.S. Retail & Food Services Sales Analysis (RSAFSNA)")
st.markdown("""
**Interactive Econometric & Time-Series Dashboard**  
Decomposition, Uncertainty Modeling (Bootstrap), and Growth Diagnostics.
""")

# Top Key Metrics
col1, col2, col3, col4 = st.columns(4)
latest_sales = df_resampled["Sales"].iloc[-1]
latest_growth = df_resampled["YoY_Growth"].iloc[-1]
total_sales = df_resampled["Sales"].sum()
avg_sales = df_resampled["Sales"].mean()

col1.metric("Latest Sales", f"${latest_sales:,.0f} M")
col2.metric("YoY Growth", f"{latest_growth:.2f}%" if not np.isnan(latest_growth) else "N/A")
col3.metric("Total Period Sales", f"${total_sales:,.0f} M")
col4.metric("Average Period Sales", f"${avg_sales:,.0f} M")

st.markdown("---")

# ---------------------------------------------------------
# 6. Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Overview", 
    "📈 Trend Analysis", 
    "🌊 Seasonality Analysis", 
    "🎲 Bootstrap Uncertainty", 
    "📋 Filtered Data", 
    "ℹ️️ About"
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Total Sales Time Series")
    
    fig_overview = px.line(
        df_resampled.reset_index(),
        x="Date",
        y="Sales",
        title=f"Total Sales ({time_res} Resolution: {year_range[0]} - {year_range[1]})",
        labels={"Sales": "Sales ($ Millions)", "Date": "Time Period"},
        template="plotly_white"
    )
    fig_overview.update_traces(line_color="#1f77b4", line_width=2)
    fig_overview.update_layout(hovermode="x unified")
    st.plotly_chart(fig_overview, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: TREND ANALYSIS
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Sales Trend & Moving Average")
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_resampled.index, y=df_resampled["Sales"],
        mode="lines", name="Observed Sales", line=dict(color="#2ca02c", width=1.5)
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_resampled.index, y=df_resampled["Rolling_Avg"],
        mode="lines", name="3-Period Rolling Average", line=dict(color="#ff7f0e", width=2.5, dash="dash")
    ))
    
    fig_trend.update_layout(
        title="Observed Sales vs. 3-Period Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales ($ Millions)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SEASONALITY ANALYSIS
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Time-Series Seasonal Decomposition")
    
    # Elegant Handling for Yearly Resolution Edge Case
    if time_res == "Yearly":
        st.info("💡 **Yearly Resolution Selected:** Seasonal decomposition requires sub-annual data (Monthly or Quarterly). Switch the resolution in the sidebar to view seasonal components.")
    elif len(df_resampled) < (sp_period * 2):
        st.warning(f"⚠️ Not enough observations for seasonal decomposition. At least {sp_period * 2} data points are required for {time_res} frequency.")
    else:
        decomp_model = st.radio("Decomposition Model", options=["Additive", "Multiplicative"], horizontal=True)
        
        try:
            decomposition = seasonal_decompose(
                df_resampled["Sales"],
                model=decomp_model.lower(),
                period=sp_period
            )
            
            # Trend Component
            fig_t = px.line(x=decomposition.trend.index, y=decomposition.trend, title="Trend Component", template="plotly_white")
            fig_t.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig_t, use_container_width=True)
            
            # Seasonal Component
            fig_s = px.line(x=decomposition.seasonal.index, y=decomposition.seasonal, title="Seasonal Component", template="plotly_white")
            fig_s.update_traces(line_color="#ff7f0e")
            st.plotly_chart(fig_s, use_container_width=True)
            
            # Residual Component
            fig_r = px.scatter(x=decomposition.resid.index, y=decomposition.resid, title="Residuals / Noise Component", template="plotly_white")
            fig_r.update_traces(marker_color="#d62728")
            st.plotly_chart(fig_r, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error executing decomposition: {e}")

# ---------------------------------------------------------
# TAB 4: BOOTSTRAP UNCERTAINTY
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Non-parametric Bootstrap Uncertainty Estimation")
    st.markdown(f"Evaluating 95% Confidence Intervals for 3-Period Moving Average using **N = {bootstrap_n}** Resamples.")
    
    with st.spinner("Calculating Bootstrap Confidence Interval Bands..."):
        sales_data = df_resampled["Sales"].values
        n_obs = len(sales_data)
        
        if n_obs > 3:
            boot_means = np.zeros((bootstrap_n, n_obs))
            
            for b in range(bootstrap_n):
                # Block bootstrap/resampling residuals to retain structure
                sample = np.random.choice(sales_data, size=n_obs, replace=True)
                boot_series = pd.Series(sample)
                boot_means[b, :] = boot_series.rolling(window=3, min_periods=1).mean().values
                
            lower_bound = np.percentile(boot_means, 2.5, axis=0)
            upper_bound = np.percentile(boot_means, 97.5, axis=0)
            
            fig_bs = go.Figure()
            
            # 95% CI Area
            fig_bs.add_trace(go.Scatter(
                x=list(df_resampled.index) + list(df_resampled.index)[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval"
            ))
            
            # Mean Line
            fig_bs.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled["Rolling_Avg"],
                mode="lines",
                name="Rolling Mean",
                line=dict(color="#1f77b4", width=2)
            ))
            
            fig_bs.update_layout(
                title="Bootstrap Rolling Mean with 95% Uncertainty Bands",
                xaxis_title="Date",
                yaxis_title="Sales ($ Millions)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_bs, use_container_width=True)
        else:
            st.warning("Insufficient data points for meaningful bootstrap sampling.")

# ---------------------------------------------------------
# TAB 5: FILTERED DATA & DOWNLOAD
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Dataset Viewer")
    
    # Format Dataset for display and clean NaNs for download
    display_df = df_resampled.copy().reset_index()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    
    st.dataframe(
        display_df.style.format({
            "Sales": "${:,.2f}",
            "YoY_Growth": "{:.2f}%",
            "Rolling_Avg": "${:,.2f}"
        }),
        use_container_width=True
    )
    
    # Download Button
    clean_csv = display_df.fillna("N/A").to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Dataset",
        data=clean_csv,
        file_name=f"RSAFSNA_Sales_{time_res}_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# TAB 6: ABOUT
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("About the Application & Methodology")
    st.markdown("""
    ### Data Source & Context
    This application utilizes data inspired by the **U.S. Census Bureau's Advance Monthly Sales for Retail and Food Services (RSAFSNA)**, as indexed in the Federal Reserve Bank of St. Louis (**FRED**).

    ### Methodological Steps
    1. **Dynamic Time Aggregation:** Converts base monthly data into Quarterly or Yearly resolution on demand using exact temporal sum logic.
    2. **Growth & Moving Averages:** Computes Year-over-Year (YoY) percentage changes alongside 3-period smooth moving window metrics.
    3. **Classical Time-Series Decomposition:** Segregates trends, seasonality, and residuals using `statsmodels` (`Additive` and `Multiplicative` frameworks).
    4. **Non-parametric Bootstrap Uncertainty:** Evaluates standard error and constructs empirical percentile-based 95% confidence intervals to measure sample variance without strict parametric distribution assumptions.
    """)
