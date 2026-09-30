import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="U.S. Retail & Food Services Sales Analysis",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD AND CLEAN DATA
# =========================================================
@st.cache_data
def load_data():
    """
    Load the original RSAFSNA monthly dataset.
    """
    df = pd.read_csv("RSAFSNA.csv")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce"
    )

    df["RSAFSNA"] = pd.to_numeric(
        df["RSAFSNA"],
        errors="coerce"
    )

    df = df.dropna(subset=["observation_date", "RSAFSNA"])

    df = df.rename(
        columns={
            "observation_date": "Date",
            "RSAFSNA": "Sales"
        }
    )

    df = (
        df[["Date", "Sales"]]
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .set_index("Date")
    )

    return df


raw_df = load_data()


# =========================================================
# 4. CACHED DATA PROCESSING
# =========================================================
@st.cache_data
def process_data(df, start_year, end_year, resolution):
    """
    Filter the data, aggregate by time resolution,
    and calculate growth and moving averages.
    """

    filtered = df[
        (df.index.year >= start_year)
        & (df.index.year <= end_year)
    ].copy()

    if resolution == "Monthly":
        result = filtered.resample("MS").sum()
        yoy_period = 12
        seasonal_period = 12

    elif resolution == "Quarterly":
        result = filtered.resample("QS").sum()
        yoy_period = 4
        seasonal_period = 4

    else:
        result = filtered.resample("YS").sum()
        yoy_period = 1
        seasonal_period = 1

    result["YoY_Growth"] = (
        result["Sales"].pct_change(yoy_period) * 100
    )

    result["Rolling_Avg"] = (
        result["Sales"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    return result, seasonal_period


# =========================================================
# 5. CACHED BOOTSTRAP
# =========================================================
@st.cache_data
def calculate_bootstrap(sales_values, n_bootstrap):
    """
    Bootstrap the mean sales level and return a 95% confidence interval.

    A fixed random seed makes the cached result reproducible.
    """

    values = np.asarray(sales_values, dtype=float)

    if len(values) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(42)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means[i] = np.mean(sample)

    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    mean_estimate = np.mean(values)

    return mean_estimate, lower, upper


# =========================================================
# 6. SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown(
    """
    Use the controls below to interactively explore
    U.S. Retail & Food Services sales.
    """
)

st.sidebar.markdown("---")


# Widget 1
time_res = st.sidebar.selectbox(
    "Time Resolution",
    options=["Monthly", "Quarterly", "Yearly"],
    index=0
)


min_year = int(raw_df.index.year.min())
max_year = int(raw_df.index.year.max())


# Widget 2
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)


st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Bootstrap Settings")


# Widget 3
bootstrap_n = st.sidebar.number_input(
    "Resampling Count (N)",
    min_value=100,
    max_value=5000,
    value=500,
    step=100
)


st.sidebar.markdown("---")

st.sidebar.caption(
    f"Original data range: "
    f"{raw_df.index.min().strftime('%b %Y')} – "
    f"{raw_df.index.max().strftime('%b %Y')}"
)


# =========================================================
# 7. PROCESS USER SELECTION
# =========================================================
df_resampled, sp_period = process_data(
    raw_df,
    year_range[0],
    year_range[1],
    time_res
)


if df_resampled.empty:
    st.error("No observations are available for the selected period.")
    st.stop()


# =========================================================
# 8. HEADER
# =========================================================
st.title("📈 U.S. Retail & Food Services Sales Analysis")

st.markdown(
    """
    **Interactive Time-Series Dashboard**

    Explore U.S. Retail & Food Services sales through
    time aggregation, growth analysis, moving averages,
    seasonal decomposition, and bootstrap uncertainty estimation.
    """
)


# =========================================================
# 9. KEY METRICS — LAYOUT WITH COLUMNS
# =========================================================
latest_sales = df_resampled["Sales"].iloc[-1]
latest_growth = df_resampled["YoY_Growth"].iloc[-1]
average_sales = df_resampled["Sales"].mean()

period_change = (
    (
        df_resampled["Sales"].iloc[-1]
        / df_resampled["Sales"].iloc[0]
        - 1
    )
    * 100
    if len(df_resampled) > 1
    else np.nan
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Latest Sales",
    f"${latest_sales:,.0f} M"
)


col2.metric(
    "Latest YoY Growth",
    (
        f"{latest_growth:.2f}%"
        if pd.notna(latest_growth)
        else "N/A"
    )
)


col3.metric(
    "Average Period Sales",
    f"${average_sales:,.0f} M"
)


col4.metric(
    "Change Across Selection",
    (
        f"{period_change:.2f}%"
        if pd.notna(period_change)
        else "N/A"
    )
)


st.markdown("---")


# =========================================================
# 10. NAVIGATION TABS
# =========================================================
tab_overview, tab_trend, tab_seasonality, tab_bootstrap, tab_data, tab_about = st.tabs(
    [
        "📊 Overview",
        "📈 Trend Analysis",
        "🌊 Seasonality",
        "🎲 Bootstrap",
        "📋 Data",
        "ℹ️ About"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab_overview:

    st.subheader("Sales Over Time")

    st.write(
        f"""
        The chart displays **{time_res.lower()} sales**
        from **{year_range[0]} to {year_range[1]}**.
        """
    )

    overview_df = df_resampled.reset_index()

    fig_overview = px.line(
        overview_df,
        x="Date",
        y="Sales",
        title=f"U.S. Retail & Food Services Sales — {time_res}",
        labels={
            "Sales": "Sales ($ Millions)",
            "Date": "Date"
        },
        template="plotly_white"
    )

    fig_overview.update_traces(
        line=dict(width=2)
    )

    fig_overview.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_overview,
        use_container_width=True
    )

    st.caption(
        "Values are based on the original RSAFSNA dataset "
        "and are aggregated according to the selected time resolution."
    )


# =========================================================
# TAB 2 — TREND ANALYSIS
# =========================================================
with tab_trend:

    st.subheader("Trend and Moving Average")

    st.write(
        """
        A 3-period moving average smooths short-term variation
        and makes the underlying direction of the series easier
        to identify.
        """
    )

    fig_trend = go.Figure()

    fig_trend.add_trace(
        go.Scatter(
            x=df_resampled.index,
            y=df_resampled["Sales"],
            mode="lines",
            name="Observed Sales"
        )
    )

    fig_trend.add_trace(
        go.Scatter(
            x=df_resampled.index,
            y=df_resampled["Rolling_Avg"],
            mode="lines",
            name="3-Period Moving Average",
            line=dict(
                width=3,
                dash="dash"
            )
        )
    )

    fig_trend.update_layout(
        title="Observed Sales vs. 3-Period Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales ($ Millions)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )


    st.subheader("Year-over-Year Growth")

    growth_df = (
        df_resampled
        .dropna(subset=["YoY_Growth"])
        .reset_index()
    )

    if not growth_df.empty:

        fig_growth = px.bar(
            growth_df,
            x="Date",
            y="YoY_Growth",
            title="Year-over-Year Sales Growth",
            labels={
                "YoY_Growth": "YoY Growth (%)",
                "Date": "Date"
            },
            template="plotly_white"
        )

        fig_growth.add_hline(
            y=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig_growth,
            use_container_width=True
        )

    else:
        st.info(
            "The selected period is too short to calculate "
            "year-over-year growth."
        )


# =========================================================
# TAB 3 — SEASONALITY
# =========================================================
with tab_seasonality:

    st.subheader("Seasonal Decomposition")

    st.write(
        """
        Seasonal decomposition separates the time series into
        **trend**, **seasonal**, and **residual** components.
        """
    )


    if time_res == "Yearly":

        st.info(
            "Seasonal decomposition requires sub-annual data. "
            "Select Monthly or Quarterly resolution."
        )


    elif len(df_resampled) < (2 * sp_period):

        st.warning(
            f"At least {2 * sp_period} observations are required "
            f"for {time_res.lower()} seasonal decomposition."
        )


    else:

        # Additional interactive widget
        decomp_model = st.radio(
            "Decomposition Model",
            options=["Additive", "Multiplicative"],
            horizontal=True
        )


        try:

            decomposition = seasonal_decompose(
                df_resampled["Sales"],
                model=decomp_model.lower(),
                period=sp_period
            )


            decomposition_df = pd.DataFrame(
                {
                    "Observed": decomposition.observed,
                    "Trend": decomposition.trend,
                    "Seasonal": decomposition.seasonal,
                    "Residual": decomposition.resid
                }
            )


            fig_decomp_trend = px.line(
                decomposition_df.reset_index(),
                x="Date",
                y="Trend",
                title="Trend Component",
                labels={
                    "Trend": "Trend",
                    "Date": "Date"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig_decomp_trend,
                use_container_width=True
            )


            fig_decomp_seasonal = px.line(
                decomposition_df.reset_index(),
                x="Date",
                y="Seasonal",
                title="Seasonal Component",
                labels={
                    "Seasonal": "Seasonal Effect",
                    "Date": "Date"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig_decomp_seasonal,
                use_container_width=True
            )


            fig_decomp_residual = px.scatter(
                decomposition_df.reset_index(),
                x="Date",
                y="Residual",
                title="Residual Component",
                labels={
                    "Residual": "Residual",
                    "Date": "Date"
                },
                template="plotly_white"
            )

            st.plotly_chart(
                fig_decomp_residual,
                use_container_width=True
            )


        except ValueError as error:

            st.error(
                f"Seasonal decomposition could not be calculated: {error}"
            )


# =========================================================
# TAB 4 — BOOTSTRAP UNCERTAINTY
# =========================================================
with tab_bootstrap:

    st.subheader("Bootstrap Uncertainty Estimation")

    st.write(
        """
        Bootstrap resampling repeatedly samples the selected
        sales observations with replacement. This provides an
        empirical estimate of uncertainty around the mean sales level.
        """
    )


    with st.spinner(
        "Calculating cached bootstrap confidence interval..."
    ):

        bootstrap_mean, lower_ci, upper_ci = calculate_bootstrap(
            df_resampled["Sales"].to_numpy(),
            int(bootstrap_n)
        )


    if pd.notna(bootstrap_mean):

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Mean Sales",
            f"${bootstrap_mean:,.0f} M"
        )

        b2.metric(
            "95% CI Lower",
            f"${lower_ci:,.0f} M"
        )

        b3.metric(
            "95% CI Upper",
            f"${upper_ci:,.0f} M"
        )


        fig_bootstrap = go.Figure()

        fig_bootstrap.add_trace(
            go.Scatter(
                x=df_resampled.index,
                y=df_resampled["Sales"],
                mode="lines",
                name="Observed Sales"
            )
        )

        fig_bootstrap.add_hline(
            y=bootstrap_mean,
            line_dash="solid",
            annotation_text="Mean"
        )

        fig_bootstrap.add_hrect(
            y0=lower_ci,
            y1=upper_ci,
            opacity=0.20,
            line_width=0,
            annotation_text="95% Bootstrap CI"
        )

        fig_bootstrap.update_layout(
            title=(
                f"Observed Sales with 95% Bootstrap Confidence "
                f"Interval (N = {bootstrap_n:,})"
            ),
            xaxis_title="Date",
            yaxis_title="Sales ($ Millions)",
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_bootstrap,
            use_container_width=True
        )

        st.caption(
            "The bootstrap computation is cached with "
            "@st.cache_data so repeated reruns with the same "
            "settings do not repeat the expensive calculation."
        )


    else:

        st.warning(
            "Not enough observations are available "
            "for bootstrap analysis."
        )


# =========================================================
# TAB 5 — DATA TABLE AND DOWNLOAD
# =========================================================
with tab_data:

    st.subheader("Filtered Dataset")

    st.write(
        f"""
        Displaying the data for **{year_range[0]}–{year_range[1]}**
        at **{time_res.lower()}** resolution.
        """
    )


    display_df = (
        df_resampled
        .copy()
        .reset_index()
    )

    display_df["Date"] = (
        display_df["Date"]
        .dt.strftime("%Y-%m-%d")
    )


    st.dataframe(
        display_df.style.format(
            {
                "Sales": "${:,.2f}",
                "YoY_Growth": "{:.2f}%",
                "Rolling_Avg": "${:,.2f}"
            },
            na_rep="N/A"
        ),
        use_container_width=True
    )


    csv_data = (
        display_df
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name=(
            f"RSAFSNA_{time_res}_"
            f"{year_range[0]}_{year_range[1]}.csv"
        ),
        mime="text/csv"
    )


# =========================================================
# TAB 6 — ABOUT
# =========================================================
with tab_about:

    st.subheader("About This Application")

    st.markdown(
        """
        ### Purpose

        This Streamlit application converts a time-series analysis
        into an interactive dashboard for exploring U.S. Retail and
        Food Services sales.

        ### Data

        The application uses the provided **RSAFSNA monthly dataset**.
        The original file contains:

        - `observation_date` — monthly observation date
        - `RSAFSNA` — Retail and Food Services sales value

        The application renames these variables to `Date` and `Sales`
        internally for readability.

        ### Analysis

        The dashboard includes:

        1. **Time filtering** using an interactive year-range slider.
        2. **Time aggregation** at monthly, quarterly, or yearly levels.
        3. **Year-over-year growth** calculations.
        4. **3-period moving averages** for trend visualization.
        5. **Seasonal decomposition** for monthly and quarterly data.
        6. **Bootstrap resampling** for uncertainty estimation.
        7. **Interactive Plotly visualizations**.
        8. **Downloadable filtered data**.

        ### Streamlit Features

        This application demonstrates:

        - Sidebar controls
        - Multiple interactive widgets
        - Columns
        - Tabs
        - Metrics
        - Interactive charts
        - `st.cache_data`
        - Download functionality
        - Responsive wide-page layout

        ### Caching

        Caching is used for both data loading and computational
        operations. The bootstrap calculation is also cached because
        repeated resampling is one of the more computationally
        expensive operations in the application.
        """
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Interactive U.S. Retail & Food Services Sales Dashboard | "
    "Built with Streamlit"
)
