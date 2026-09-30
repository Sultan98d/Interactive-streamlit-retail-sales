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
    page_title="U.S. Retail Sales Time-Series Analysis",
    layout="wide"
)

st.title("U.S. Retail Sales Time-Series Analysis")

st.write(
    "This app analyzes U.S. retail and food services sales from 1992 to 2026. "
    "The data are from the U.S. Census Bureau via FRED and are not seasonally adjusted."
)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

df = pd.read_csv("RSAFSNA.csv")

# Convert date column to datetime
df["observation_date"] = pd.to_datetime(
    df["observation_date"],
    errors="coerce"
)

# Convert sales values to numeric
df["RSAFSNA"] = pd.to_numeric(
    df["RSAFSNA"],
    errors="coerce"
)

# Remove invalid rows
df = df.dropna(
    subset=["observation_date", "RSAFSNA"]
)

# Use date as index
df = df.set_index("observation_date")

# Sort dates
df = df.sort_index()


# =========================================================
# 1. TIME RESOLUTION
# =========================================================

st.header("1. Time Resolution")

resolution = st.selectbox(
    "Select a time resolution:",
    ["Monthly", "Quarterly", "Yearly"]
)

if resolution == "Monthly":

    data = (
        df["RSAFSNA"]
        .resample("MS")
        .mean()
        .dropna()
    )

    window = 12
    window_text = "12 months"

elif resolution == "Quarterly":

    data = (
        df["RSAFSNA"]
        .resample("QS")
        .mean()
        .dropna()
    )

    window = 4
    window_text = "4 quarters"

else:

    data = (
        df["RSAFSNA"]
        .resample("YS")
        .mean()
        .dropna()
    )

    window = 1
    window_text = "1 year"


st.caption(
    f"Current resolution: {resolution}. "
    f"Trend and uncertainty window: {window_text}."
)


# =========================================================
# 2. RETAIL SALES OVER TIME
# =========================================================

st.header("2. Retail Sales Over Time")

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

# Zero baseline for honest visual comparison
ax.set_ylim(bottom=0)

ax.grid(alpha=0.3)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# INTERACTIVE DATE-RANGE BRUSH
# =========================================================

st.subheader("Interactive Date-Range Brush")

brush_df = data.reset_index()
brush_df.columns = ["Date", "Sales"]

# Interval selection on the x-axis
brush = alt.selection_interval(
    encodings=["x"],
    name="DateRange"
)

# Main chart:
# its x-axis domain follows the range selected below
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

# Lower chart:
# drag horizontally here to choose the date range
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
        title="Click and drag inside this chart to select a date range"
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
    "Click and drag horizontally inside the lower chart to select a date range. "
    "The upper chart automatically zooms to the selected period."
)


# =========================================================
# 3. TREND
# =========================================================

st.header("3. Trend")

trend = data.rolling(
    window=window,
    center=True
).mean()

fig, ax = plt.subplots(figsize=(12, 5))

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

    st.write(
        "A 12-month rolling mean is used because the data are monthly. "
        "The window covers one complete annual cycle, which reduces short-term "
        "seasonal variation and makes the long-term trend easier to see."
    )

elif resolution == "Quarterly":

    st.write(
        "A 4-quarter rolling mean is used because four quarters make one year. "
        "This reduces short-term variation while keeping the long-term trend visible."
    )

else:

    st.write(
        "At yearly resolution, each observation already summarizes one complete year. "
        "Therefore, a one-year window is used instead of adding extra multi-year smoothing."
    )


# =========================================================
# 4. SEASONALITY
# =========================================================

st.header("4. Seasonality")

# Keep monthly data for annual seasonality
monthly = (
    df["RSAFSNA"]
    .resample("MS")
    .mean()
    .dropna()
)

# Additive decomposition
additive = seasonal_decompose(
    monthly,
    model="additive",
    period=12
)

# Average seasonal effect for each calendar month
seasonal_cycle = (
    additive.seasonal
    .groupby(additive.seasonal.index.month)
    .mean()
)

month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

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
    "Seasonal decomposition is performed on the monthly series using period = 12 "
    "because twelve monthly observations make one annual cycle. "
    "This chart summarizes the recurring January-to-December seasonal pattern."
)


# =========================================================
# ADDITIVE DECOMPOSITION
# =========================================================

st.subheader("Additive Seasonal Decomposition")

fig = additive.plot()

fig.set_size_inches(
    12,
    8
)

st.pyplot(fig)

plt.close(fig)

st.caption(
    "The monthly series is separated into observed, trend, seasonal, "
    "and residual components."
)


# =========================================================
# OPTIONAL EXTENSION:
# ADDITIVE VS. MULTIPLICATIVE
# =========================================================

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

# Additive residuals
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


# Multiplicative residuals
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

axes[1].set_ylabel(
    "Residual Ratio"
)

axes[1].grid(alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

st.write(
    "The additive model assumes that the size of the seasonal effect stays "
    "roughly constant over time. The multiplicative model assumes that the "
    "seasonal effect changes in proportion to the level of sales. "
    "The residual plots allow the two approaches to be compared."
)


# =========================================================
# 5. UNCERTAINTY
# =========================================================

st.header("5. Uncertainty")

st.write(
    "This analysis uses a bootstrap 95% confidence interval for the rolling mean "
    "instead of a ±2 standard-deviation spread band."
)


# =========================================================
# BOOTSTRAP FUNCTION
# =========================================================

def bootstrap_ci(
    values,
    n_boot=1000,
    confidence=0.95
):

    values = np.asarray(values)

    values = values[
        ~np.isnan(values)
    ]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(42)

    samples = rng.choice(
        values,
        size=(
            n_boot,
            len(values)
        ),
        replace=True
    )

    means = samples.mean(axis=1)

    alpha = 1 - confidence

    lower = np.quantile(
        means,
        alpha / 2
    )

    upper = np.quantile(
        means,
        1 - alpha / 2
    )

    return lower, upper


# =========================================================
# ROLLING MEAN
# =========================================================

rolling_mean = data.rolling(
    window=window
).mean()

lower_ci = pd.Series(
    np.nan,
    index=data.index
)

upper_ci = pd.Series(
    np.nan,
    index=data.index
)


# =========================================================
# CALCULATE BOOTSTRAP INTERVALS
# =========================================================

for i in range(
    window - 1,
    len(data)
):

    values = data.iloc[
        i - window + 1:
        i + 1
    ].values

    low, high = bootstrap_ci(values)

    lower_ci.iloc[i] = low
    upper_ci.iloc[i] = high


# =========================================================
# UNCERTAINTY CHART
# =========================================================

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

ax.set_ylabel(
    "Millions of Dollars"
)

ax.set_ylim(bottom=0)

ax.legend()

ax.grid(alpha=0.3)

st.pyplot(fig)

plt.close(fig)


st.write(
    "For each rolling window, the observations are resampled with replacement "
    "1,000 times. A mean is calculated for every bootstrap sample. "
    "The 2.5th and 97.5th percentiles of the bootstrap means form the 95% "
    "confidence interval. This interval represents uncertainty in the estimated "
    "local mean and is different from a ±2 standard-deviation band, which "
    "describes the spread of individual observations."
)


# =========================================================
# 6. TEMPORAL HONESTY
# =========================================================

st.header("6. Temporal-Honesty Note")

st.write(
    "The original data are monthly. The resolution selector allows the same "
    "series to be viewed at monthly, quarterly, or yearly resolution, and the "
    "selected resolution is stated directly on the chart. Quarterly and yearly "
    "values are calculated using the mean within each period. The full 1992–2026 "
    "time range is retained so the long-term trend is shown in context. "
    "For monthly and quarterly views, the rolling window covers one complete "
    "year and the window size is explicitly stated. At yearly resolution, each "
    "point already summarizes one full year, so no additional multi-year "
    "smoothing is introduced. Sales charts use a zero-based y-axis to avoid "
    "visually exaggerating changes. The charts use consistent aspect ratios "
    "and no dual y-axes. Monthly resolution is retained for seasonal "
    "decomposition because it preserves the 12-month annual cycle."
)


# =========================================================
# 7. SUMMARY
# =========================================================

st.header("7. Summary")

st.write(
    "U.S. retail sales show a strong long-term upward trend together with "
    "a recurring annual seasonal pattern. Resampling demonstrates how temporal "
    "resolution changes the appearance of the series. Rolling averages make "
    "the long-term trend easier to see, while seasonal decomposition separates "
    "trend, seasonality, and residual variation. The bootstrap confidence "
    "interval communicates uncertainty in the estimated rolling mean."
)


# =========================================================
# DATA SOURCE
# =========================================================

st.caption(
    "Data source: U.S. Census Bureau via FRED — "
    "Advance Retail Sales: Retail Trade and Food Services (RSAFSNA), "
    "Monthly, Not Seasonally Adjusted."
)