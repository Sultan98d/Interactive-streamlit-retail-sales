import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="U.S. Retail Sales Interactive Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("U.S. Retail Sales Interactive Dashboard")

st.write(
    "Explore U.S. retail and food services sales over time. "
    "Use the sidebar controls to change the time resolution, "
    "filter the analysis period, and customize the visualization."
)


# =========================================================
# LOAD AND PREPARE DATA — CACHED
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("RSAFSNA.csv")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce"
    )

    df["RSAFSNA"] = pd.to_numeric(
        df["RSAFSNA"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["observation_date", "RSAFSNA"]
    )

    df = df.set_index("observation_date")
    df = df.sort_index()

    return df


df = load_data()


# =========================================================
# SIDEBAR + WIDGETS
# =========================================================

st.sidebar.header("Dashboard Controls")

st.sidebar.write(
    "Use these controls to customize the time-series analysis."
)

# Widget 1
resolution = st.sidebar.selectbox(
    "Time resolution",
    ["Monthly", "Quarterly", "Yearly"],
    key="resolution"
)

min_year = int(df.index.year.min())
max_year = int(df.index.year.max())

# Widget 2
year_range = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    key="year_range"
)

# Widget 3
show_points = st.sidebar.checkbox(
    "Show observations on trend chart",
    value=False,
    key="show_points"
)

st.sidebar.divider()

st.sidebar.caption(
    "Data source: U.S. Census Bureau via FRED (RSAFSNA)."
)


# =========================================================
# SESSION STATE
# =========================================================

if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

if "previous_settings" not in st.session_state:
    st.session_state.previous_settings = (
        resolution,
        year_range,
        show_points
    )

current_settings = (
    resolution,
    year_range,
    show_points
)

if current_settings != st.session_state.previous_settings:
    st.session_state.interaction_count += 1
    st.session_state.previous_settings = current_settings


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df.index.year >= year_range[0]) &
    (df.index.year <= year_range[1])
].copy()

if resolution == "Monthly":

    data = (
        filtered_df["RSAFSNA"]
        .resample("MS")
        .mean()
        .dropna()
    )

    window = 12
    window_text = "12 months"

elif resolution == "Quarterly":

    data = (
        filtered_df["RSAFSNA"]
        .resample("QS")
        .mean()
        .dropna()
    )

    window = 4
    window_text = "4 quarters"

else:

    data = (
        filtered_df["RSAFSNA"]
        .resample("YS")
        .mean()
        .dropna()
    )

    window = 1
    window_text = "1 year"


# =========================================================
# SUMMARY METRICS — COLUMNS
# =========================================================

st.subheader("Current Selection")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Resolution",
    resolution
)

col2.metric(
    "Start Year",
    year_range[0]
)

col3.metric(
    "End Year",
    year_range[1]
)

col4.metric(
    "Observations",
    len(data)
)

st.caption(
    f"Trend and uncertainty window: {window_text}. "
    f"Dashboard settings changed {st.session_state.interaction_count} time(s) "
    "during this session."
)


# =========================================================
# TABS — LAYOUT
# =========================================================

tab_overview, tab_trend, tab_seasonality, tab_uncertainty, tab_data, tab_about = st.tabs(
    [
        "Overview",
        "Trend",
        "Seasonality",
        "Uncertainty",
        "Data",
        "About"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab_overview:

    st.header("Retail Sales Over Time")

    st.write(
        f"The chart below displays U.S. retail sales using "
        f"{resolution.lower()} resolution from {year_range[0]} "
        f"through {year_range[1]}."
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data.index,
        data.values
    )

    ax.set_title(
        f"U.S. Retail Sales — {resolution} Resolution"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Millions of Dollars")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


    # -----------------------------------------------------
    # INTERACTIVE DATE-RANGE BRUSH
    # -----------------------------------------------------

    st.subheader("Interactive Date-Range Brush")

    brush_df = data.reset_index()
    brush_df.columns = ["Date", "Sales"]

    brush = alt.selection_interval(
        encodings=["x"],
        name="DateRange"
    )

    main_chart = (
        alt.Chart(brush_df)
        .mark_line()
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                scale=alt.Scale(domain=brush)
            ),
            y=alt.Y(
                "Sales:Q",
                title="Millions of Dollars",
                scale=alt.Scale(zero=True)
            ),
            tooltip=[
                alt.Tooltip(
                    "Date:T",
                    title="Date"
                ),
                alt.Tooltip(
                    "Sales:Q",
                    title="Sales",
                    format=",.0f"
                )
            ]
        )
        .properties(
            height=350,
            title="Selected Date Range"
        )
    )

    overview_chart = (
        alt.Chart(brush_df)
        .mark_line()
        .encode(
            x=alt.X(
                "Date:T",
                title="Date"
            ),
            y=alt.Y(
                "Sales:Q",
                title="Millions of Dollars",
                scale=alt.Scale(zero=True)
            )
        )
        .add_params(brush)
        .properties(
            height=120,
            title="Click and drag to select a date range"
        )
    )

    interactive_chart = alt.vconcat(
        main_chart,
        overview_chart
    )

    st.altair_chart(
        interactive_chart,
        use_container_width=True
    )

    st.caption(
        "Click and drag horizontally inside the lower chart. "
        "The upper chart automatically zooms to the selected period."
    )


# =========================================================
# TAB 2 — TREND
# =========================================================

with tab_trend:

    st.header("Trend Analysis")

    trend = data.rolling(
        window=window,
        center=True
    ).mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    if show_points:

        ax.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            alpha=0.35,
            label=f"{resolution} Sales"
        )

    else:

        ax.plot(
            data.index,
            data.values,
            alpha=0.35,
            label=f"{resolution} Sales"
        )

    ax.plot(
        trend.index,
        trend.values,
        linewidth=2.5,
        label=f"Rolling Trend ({window_text})"
    )

    ax.set_title("U.S. Retail Sales Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Millions of Dollars")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    if resolution == "Monthly":

        st.info(
            "A 12-month rolling mean is used because the data are monthly. "
            "The window covers one complete annual cycle and reduces "
            "short-term seasonal variation."
        )

    elif resolution == "Quarterly":

        st.info(
            "A 4-quarter rolling mean is used because four quarters "
            "make one year. This reduces short-term variation while "
            "keeping the long-term trend visible."
        )

    else:

        st.info(
            "At yearly resolution, each observation already summarizes "
            "one complete year. Therefore, a one-year window is used."
        )


# =========================================================
# TAB 3 — SEASONALITY
# =========================================================

with tab_seasonality:

    st.header("Seasonality")

    monthly = (
        filtered_df["RSAFSNA"]
        .resample("MS")
        .mean()
        .dropna()
    )

    if len(monthly) >= 24:

        additive = seasonal_decompose(
            monthly,
            model="additive",
            period=12
        )

        seasonal_cycle = (
            additive.seasonal
            .groupby(additive.seasonal.index.month)
            .mean()
        )

        month_names = [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]

        st.subheader("Average Annual Seasonal Pattern")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            month_names,
            seasonal_cycle.values,
            marker="o"
        )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.set_title(
            "Average Annual Seasonal Pattern"
        )

        ax.set_xlabel("Month")

        ax.set_ylabel(
            "Seasonal Effect (Millions of Dollars)"
        )

        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.write(
            "Seasonal decomposition is performed on the monthly series "
            "using period = 12 because twelve monthly observations make "
            "one annual cycle."
        )


        # -------------------------------------------------
        # ADDITIVE DECOMPOSITION
        # -------------------------------------------------

        st.subheader("Additive Seasonal Decomposition")

        fig = additive.plot()
        fig.set_size_inches(12, 8)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The monthly series is separated into observed, trend, "
            "seasonal, and residual components."
        )


        # -------------------------------------------------
        # ADDITIVE VS MULTIPLICATIVE
        # -------------------------------------------------

        st.subheader(
            "Additive vs. Multiplicative Seasonal Decomposition"
        )

        multiplicative = seasonal_decompose(
            monthly,
            model="multiplicative",
            period=12
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True
        )

        axes[0].plot(
            additive.resid.index,
            additive.resid.values
        )

        axes[0].axhline(
            0,
            linewidth=1
        )

        axes[0].set_title(
            "Additive Model Residuals"
        )

        axes[0].set_ylabel("Residual")
        axes[0].grid(alpha=0.3)

        axes[1].plot(
            multiplicative.resid.index,
            multiplicative.resid.values
        )

        axes[1].axhline(
            1,
            linewidth=1
        )

        axes[1].set_title(
            "Multiplicative Model Residuals"
        )

        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Residual Ratio")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.write(
            "The additive model assumes that the seasonal effect stays "
            "roughly constant over time. The multiplicative model assumes "
            "that the seasonal effect changes in proportion to the level "
            "of sales."
        )

    else:

        st.warning(
            "Select a range containing at least two full years "
            "to display seasonal decomposition."
        )


# =========================================================
# CACHED BOOTSTRAP FUNCTION
# =========================================================

@st.cache_data
def calculate_bootstrap_intervals(
    values_tuple,
    window,
    n_boot=1000,
    confidence=0.95
):

    values_array = np.asarray(
        values_tuple,
        dtype=float
    )

    lower = np.full(
        len(values_array),
        np.nan
    )

    upper = np.full(
        len(values_array),
        np.nan
    )

    rng = np.random.default_rng(42)

    alpha = 1 - confidence

    for i in range(
        window - 1,
        len(values_array)
    ):

        current_values = values_array[
            i - window + 1:
            i + 1
        ]

        current_values = current_values[
            ~np.isnan(current_values)
        ]

        if len(current_values) < 2:
            continue

        samples = rng.choice(
            current_values,
            size=(
                n_boot,
                len(current_values)
            ),
            replace=True
        )

        means = samples.mean(axis=1)

        lower[i] = np.quantile(
            means,
            alpha / 2
        )

        upper[i] = np.quantile(
            means,
            1 - alpha / 2
        )

    return lower, upper


# =========================================================
# TAB 4 — UNCERTAINTY
# =========================================================

with tab_uncertainty:

    st.header("Uncertainty")

    st.write(
        "This analysis uses a bootstrap 95% confidence interval "
        "for the rolling mean."
    )

    rolling_mean = data.rolling(
        window=window
    ).mean()

    lower_values, upper_values = calculate_bootstrap_intervals(
        tuple(data.values),
        window
    )

    lower_ci = pd.Series(
        lower_values,
        index=data.index
    )

    upper_ci = pd.Series(
        upper_values,
        index=data.index
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data.index,
        data.values,
        alpha=0.30,
        label=f"{resolution} Sales"
    )

    ax.plot(
        rolling_mean.index,
        rolling_mean.values,
        linewidth=2.5,
        label=f"Rolling Mean ({window_text})"
    )

    ax.fill_between(
        data.index,
        lower_ci,
        upper_ci,
        alpha=0.25,
        label="Bootstrap 95% CI for Rolling Mean"
    )

    ax.set_title(
        "Retail Sales with Bootstrap 95% Confidence Interval"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Millions of Dollars")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "For each rolling window, observations are resampled with "
        "replacement 1,000 times. The 2.5th and 97.5th percentiles "
        "of the bootstrap means form the 95% confidence interval. "
        "The calculation is cached so Streamlit does not repeat the "
        "same expensive computation unnecessarily."
    )


# =========================================================
# TAB 5 — DATA
# =========================================================

with tab_data:

    st.header("Filtered Data")

    display_df = data.reset_index()
    display_df.columns = ["Date", "Sales"]

    st.dataframe(
        display_df,
        use_container_width=True
    )

    csv = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_retail_sales.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 6 — ABOUT
# =========================================================

with tab_about:

    st.header("About This Dashboard")

    st.write(
        "This interactive Streamlit dashboard extends a time-series "
        "analysis of U.S. retail and food services sales. Users can "
        "change the temporal resolution, select a year range, and "
        "control how observations appear in the trend visualization."
    )

    st.subheader("Analysis")

    st.write(
        "The dashboard includes the original time-series visualization, "
        "rolling trend analysis, seasonal decomposition, an interactive "
        "date-range brush, and bootstrap uncertainty intervals."
    )

    st.subheader("Temporal-Honesty Note")

    st.write(
        "The original data are monthly. Quarterly and yearly values are "
        "calculated using the mean within each period. For monthly and "
        "quarterly views, the rolling window covers one complete year. "
        "At yearly resolution, each point already summarizes one full "
        "year. Sales charts use a zero-based y-axis to avoid visually "
        "exaggerating changes. Monthly resolution is retained for "
        "seasonal decomposition because it preserves the 12-month "
        "annual cycle."
    )

    st.subheader("Caching")

    st.write(
        "The application uses Streamlit caching when loading and "
        "preparing the dataset and when calculating bootstrap confidence "
        "intervals. This prevents expensive operations from being "
        "repeated unnecessarily during app reruns."
    )

    st.subheader("Data Source")

    st.write(
        "U.S. Census Bureau via FRED — Advance Retail Sales: "
        "Retail Trade and Food Services (RSAFSNA), Monthly, "
        "Not Seasonally Adjusted."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "U.S. Retail Sales Interactive Dashboard | "
    "Built with Streamlit"
)
