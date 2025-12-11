# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:08:14 2025

@author: epicx
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:35:17 2025

@author: epicx

EDA plotting utilities for TLC + weather hourly data.

Usage (from repo root):

    python -m code.eda_plotting --input data/processed/tlc/your_file.parquet
    python -m code.eda_plotting --input data/processed/tlc/your_file.parquet --save-plots

If --save-plots is set, figures are written to:
    <PROJECT_ROOT>/plots
which on your PC is:
    C:\\Users\\epicx\\Projects\\ubercasestudy\\plots
"""

from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

from config import PROJECT_ROOT, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------

PLOTS_DIR = PROJECT_ROOT / "plots"

DEFAULT_TIME_COL = "datetime_hour"

# Time-series we actually want to see over time
DEFAULT_TS_COLS = [
    # volumes / counts
    "request_count",
    "pickup_count",
    "dropoff_count",

    # wait times (seconds)
    "wait_time_sec_mean",
    "wait_time_sec_median",

    # distance / time / speed
    "trip_miles_mean",
    "trip_miles_median",
    "trip_time_mean",
    "trip_time_median",
    "avg_speed_mph",
    "median_speed_mph",

    # economics (per trip averages)
    "avg_base_passenger_fare",
    "avg_subtotal_fare",
    "avg_rider_total",
    "avg_tips",
    "avg_driver_pay",
    "avg_driver_pay_with_tips",
    "driver_pay_pct_of_base_fare",

    # weather
    "temp_f_mean",
    "dewpoint_f_mean",
    "wind_speed_mph_mean",
    "precip_1h_mm_total",
    "relative_humidity",
    "wind_chill_f",
    "heat_index_f",
]

# Narrow but rich set for correlations (add sums and a few more weather stats)
DEFAULT_CORR_COLS = [
    # volumes
    "request_count",
    "pickup_count",
    "dropoff_count",

    # totals
    "trip_miles_sum",
    "trip_time_sum",
    "rider_total_sum",
    "driver_pay_sum",

    # per-trip economics
    "avg_base_passenger_fare",
    "avg_subtotal_fare",
    "avg_rider_total",
    "avg_tips",
    "avg_driver_pay",
    "avg_driver_pay_with_tips",
    "driver_pay_pct_of_base_fare",

    # performance / congestion proxies
    "avg_speed_mph",
    "median_speed_mph",
    "wait_time_sec_mean",
    "wait_time_sec_median",

    # weather core
    "temp_f_mean",
    "dewpoint_f_mean",
    "wind_speed_mph_mean",
    "precip_1h_mm_total",
    "relative_humidity",
    "wind_chill_f",
    "heat_index_f",
]

# (x, y) pairs for scatter plots aimed at demand + economics vs weather/ops
DEFAULT_SCATTER_PAIRS = [
    ("temp_f_mean", "request_count"),
    ("wind_chill_f", "request_count"),
    ("heat_index_f", "request_count"),
    ("precip_1h_mm_total", "request_count"),
    ("relative_humidity", "request_count"),

    ("temp_f_mean", "avg_base_passenger_fare"),
    ("wind_chill_f", "avg_base_passenger_fare"),
    ("heat_index_f", "avg_base_passenger_fare"),
    ("precip_1h_mm_total", "avg_base_passenger_fare"),

    ("temp_f_mean", "avg_driver_pay"),
    ("wind_chill_f", "avg_driver_pay"),
    ("heat_index_f", "avg_driver_pay"),
    ("precip_1h_mm_total", "avg_driver_pay"),
]

# Hours to drop when curfew=True (LaGuardia midnight–6am)
CURFEW_HOURS = set(range(2, 7))  # 2,3,4,5,6

PRECIP_COL = "precip_1h_mm_total"

DAY_COLOR_MAP = {
    0: "tab:blue",    # Monday
    1: "tab:orange",  # Tuesday
    2: "tab:green",   # Wednesday
    3: "tab:red",     # Thursday
    4: "tab:purple",  # Friday
    5: "tab:brown",   # Saturday
    6: "tab:pink",    # Sunday
}

# ---------------------------------------------------------------------------
# IO / helpers
# ---------------------------------------------------------------------------


def _ensure_out_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        out_dir = PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_input(input_path: Path) -> pd.DataFrame:
    """Read a processed parquet file."""
    df = pd.read_parquet(input_path)
    return df


def _make_precip_buckets(s: pd.Series) -> pd.Series:
    """
    Bucket precip amounts (in mm) into:
        "0", "(0,1]", "(1,2]", ..., "(49,50]", "(50,∞]".
    Missing treated as 0.
    """
    s = s.fillna(0).clip(lower=0)

    zero_mask = (s == 0)
    pos = s[~zero_mask]

    # bins: (0,1], (1,2], . (49,50], (50, inf]
    edges = list(range(0, 51)) + [np.inf]
    pos_binned = pd.cut(pos, bins=edges, right=True, include_lowest=False)

    # Build labels
    labels = ["0"]
    labels += [f"({i},{i+1}]" for i in range(0, 50)]
    labels.append("(50,∞]")

    bucket = pd.Series(index=s.index, dtype="object")
    bucket[zero_mask] = "0"

    # Map interval -> label for positives
    interval_to_label = {}
    # intervals from 0–50 and 50+; len(pos_binned.cat.categories) should be 51
    cats = pos_binned.cat.categories
    for i, cat in enumerate(cats):
        if i < 50:
            interval_to_label[cat] = f"({i},{i+1}]"
        else:
            interval_to_label[cat] = "(50,∞]"

    bucket[~zero_mask] = pos_binned.map(interval_to_label)
    bucket = bucket.astype("category")
    bucket = bucket.cat.set_categories(labels, ordered=True)

    return bucket


def _plot_simple_hist(
    series: pd.Series,
    title: str,
    xlabel: str,
    filename: str | None,
    save: bool,
    out_dir: Path | None,
    bin_width: float,
) -> None:
    s = series.dropna()
    if s.empty:
        print(f"No data for {title}; skipping.")
        return

    mn = np.floor(s.min() / bin_width) * bin_width
    mx = np.ceil(s.max() / bin_width) * bin_width
    bins = np.arange(mn, mx + bin_width, bin_width)

    median_val = s.median()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    # numeric median
    ax.axvline(
        median_val,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Median = {median_val:.1f}"
    )
    ax.legend()

    if save and filename is not None:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _scatter_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    save: bool,
    out_dir: Path | None,
) -> None:
    """Generic scatter helper for core diagnostics."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {title}: missing {x_col} or {y_col}.")
        return

    s = df[[x_col, y_col]].dropna()
    if s.empty:
        print(f"Skipping {title}: no non-null data.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(s[x_col], s[y_col], alpha=0.4, s=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _normalize_dow(dow) -> int:
    """
    Convert various weekday representations to pandas-style dayofweek 0–6.

    Accepts:
        - integer 0–6
        - string names like "Mon", "Monday", case-insensitive
    """
    if isinstance(dow, int):
        if 0 <= dow <= 6:
            return dow
        raise ValueError("Day of week integer must be in [0, 6].")

    if isinstance(dow, str):
        dow = dow.strip().lower()
        mapping = {
            "mon": 0,
            "monday": 0,
            "tue": 1,
            "tues": 1,
            "tuesday": 1,
            "wed": 2,
            "weds": 2,
            "wednesday": 2,
            "thu": 3,
            "thur": 3,
            "thurs": 3,
            "thursday": 3,
            "fri": 4,
            "friday": 4,
            "sat": 5,
            "saturday": 5,
            "sun": 6,
            "sunday": 6,
        }
        if dow not in mapping:
            raise ValueError(f"Unrecognized day-of-week string: {dow}")
        return mapping[dow]

    raise TypeError("Day-of-week must be int 0–6 or a weekday string.")


def _aggregate_by_hour_of_week(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = DEFAULT_TIME_COL,
    agg: str = "mean",
    curfew: bool = False,
) -> pd.DataFrame:
    """
    Aggregate a metric by hour-of-week (0=Mon 00:00, ..., 167=Sun 23:00).

    Returns a DataFrame with:
        hour_of_week  (int 0–167)
        value         (aggregated metric)
        dow           (0=Mon,...,6=Sun)
        hour          (0–23)
    """
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing {time_col} or {value_col} in dataframe.")

    d = df.copy()

    # Optional curfew filter (drop 2–6AM across all days)
    if curfew:
        if "request_hour" in d.columns:
            d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()
        else:
            # Fallback to hour from time_col
            d["hour"] = d[time_col].dt.hour
            d = d[~d["hour"].isin(CURFEW_HOURS)].copy()

    d["dow"] = d[time_col].dt.dayofweek  # 0=Mon,...,6=Sun
    d["hour"] = d[time_col].dt.hour
    d["hour_of_week"] = d["dow"] * 24 + d["hour"]

    grouped = (
        d.groupby("hour_of_week", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: "value"})
    )

    # Reindex to full 0–167 grid
    all_hours = pd.DataFrame({"hour_of_week": np.arange(168)})
    grouped = all_hours.merge(grouped, on="hour_of_week", how="left")
    grouped["dow"] = grouped["hour_of_week"] // 24
    grouped["hour"] = grouped["hour_of_week"] % 24

    return grouped

# ---------------------------------------------------------------------------
# Simple TS, scatter & correlation plots
# ---------------------------------------------------------------------------


def plot_time_series(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    time_col: str = DEFAULT_TIME_COL,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Quick time series plots of all requested metrics.
    """
    if cols is None:
        cols = DEFAULT_TS_COLS

    # Filter to existing cols
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("No requested columns found for time-series plotting.")
        return

    if time_col not in df.columns:
        print(f"Missing {time_col}; skipping time-series plots.")
        return

    d = df[[time_col] + cols].sort_values(time_col).copy()

    for col in cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(d[time_col], d[col])
        ax.set_title(col)
        ax.set_xlabel(time_col)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        if save:
            if out_dir is None:
                out_dir = PLOTS_DIR
            out_dir = _ensure_out_dir(out_dir)
            fname = f"ts_{col}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


def plot_scatter_pairs(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Plot simple scatter charts for the configured pairs of (x, y) cols.
    """
    if pairs is None:
        pairs = DEFAULT_SCATTER_PAIRS

    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        fname = f"scatter_{y_col}_vs_{x_col}.png"
        _scatter_xy(
            df=df,
            x_col=x_col,
            y_col=y_col,
            title=f"{y_col} vs {x_col}",
            xlabel=x_col,
            ylabel=y_col,
            filename=fname,
            save=save,
            out_dir=out_dir,
        )


def plot_corr_heatmap(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Correlation heatmap for a narrow set of metrics + weather.
    """
    if cols is None:
        cols = DEFAULT_CORR_COLS

    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        print("Not enough columns for correlation heatmap; skipping.")
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    fig.colorbar(cax, ax=ax)
    ax.set_title("Correlation heatmap")

    fig.tight_layout()

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fig.savefig(out_dir / "corr_heatmap.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hour-of-day distributions, precip / wait-time histograms
# ---------------------------------------------------------------------------


def plot_requests_by_hour_of_day(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    count_col: str = "request_count",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Requests by hour-of-day across the full sample.

    Plots:
      - mean requests by hour-of-day
      - optionally drop curfew hours (2–6) before aggregating
    """
    if time_col not in df.columns or count_col not in df.columns:
        print(f"Missing {time_col} or {count_col}; skipping hour-of-day plot.")
        return

    d = df.copy()
    if curfew and "request_hour" in d.columns:
        d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()

    d["hour"] = d[time_col].dt.hour

    grouped = (
        d.groupby("hour", as_index=False)[count_col]
        .mean()
        .rename(columns={count_col: "avg_requests"})
        .sort_values("hour")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped["hour"], grouped["avg_requests"], marker="o")
    ax.set_title("Avg requests by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg requests")
    ax.set_xticks(range(24))
    ax.grid(True, axis="y", alpha=0.3)

    if curfew:
        ax.set_title("Avg requests by hour of day (curfew hours removed)")

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fname = "requests_by_hour_of_day.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_precip_distributions(
    df: pd.DataFrame,
    precip_col: str = PRECIP_COL,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Histograms and bucketed distribution for the 1-hour precip column.
    """
    if precip_col not in df.columns:
        print(f"Missing {precip_col}; skipping precip distributions.")
        return

    s = df[precip_col]

    # Simple histogram (mm)
    _plot_simple_hist(
        s,
        title="Precipitation (1h) distribution",
        xlabel="Precipitation (mm)",
        filename="precip_1h_hist.png",
        save=save,
        out_dir=out_dir,
        bin_width=1.0,
    )

    # Bucketed categories
    buckets = _make_precip_buckets(s)
    counts = buckets.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Precipitation buckets (1h)")
    ax.set_xlabel("Bucket (mm)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fig.savefig(out_dir / "precip_1h_buckets.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_weather_and_waittime_histograms(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Basic histograms for:
      - wait_time_sec_mean, wait_time_sec_median
      - temp_f_mean, wind_chill_f, heat_index_f
    """
    cols = [
        ("wait_time_sec_mean", "Mean wait time (sec)", 30.0),
        ("wait_time_sec_median", "Median wait time (sec)", 30.0),
        ("temp_f_mean", "Mean temperature (F)", 2.0),
        ("wind_chill_f", "Wind chill (F)", 2.0),
        ("heat_index_f", "Heat index (F)", 2.0),
    ]

    for col, xlabel, bin_width in cols:
        if col not in df.columns:
            continue
        _plot_simple_hist(
            df[col],
            title=f"Distribution of {col}",
            xlabel=xlabel,
            filename=f"hist_{col}.png",
            save=save,
            out_dir=out_dir,
            bin_width=bin_width,
        )


# ---------------------------------------------------------------------------
# Hour-of-week generic cores and wrappers
# ---------------------------------------------------------------------------


def plot_requests_by_hour_of_week(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    count_col: str = "request_count",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Backward-compatible wrapper that uses the generic multi-series demand plot
    but only shows realized demand + hourly-of-day baseline (no model line).
    """
    if time_col != DEFAULT_TIME_COL:
        df = df.copy()
        df[DEFAULT_TIME_COL] = df[time_col]

    plot_demand_hour_of_week_multi(
        df=df,
        realized_col=count_col,
        est_col=None,           # no model line in this wrapper
        est_label="Model estimate",
        time_col=DEFAULT_TIME_COL,
        save=save,
        out_dir=out_dir,
        curfew=curfew,
    )


def plot_metric_by_hour_of_week(
    df: pd.DataFrame,
    metric_col: str,
    agg: str = "mean",
    time_col: str = DEFAULT_TIME_COL,
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
    ylabel: str | None = None,
    title: str | None = None,
    filename: str | None = None,
    line_color: str = "tab:blue",
) -> None:
    """
    Clean hour-of-week plot: 168-point average pattern for a single metric.

    - x-axis: 0–167 with ticks centered in each day (Mon,...,Sun)
    - y-axis: per-hour average (agg) of metric_col
    - light shading for Saturday+Sunday only
    """
    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    grouped = _aggregate_by_hour_of_week(
        df=df,
        value_col=metric_col,
        time_col=time_col,
        agg=agg,
        curfew=curfew,
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        grouped["hour_of_week"],
        grouped["value"],
        color=line_color,
        linewidth=1.4,
        marker="o",
        markersize=3,
    )

    if title is None:
        title = f"{metric_col} by hour of week"
    ax.set_title(title)

    if ylabel is None:
        ylabel = metric_col
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Hour of week (0=Mon 00:00)")

    # X-ticks: one per day, centered at hour 12
    xticks = [24 * d + 12 for d in range(7)]
    xtick_labels = [calendar.day_name[d] for d in range(7)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)

    # Weekend shading: Saturday + Sunday
    sat_start = 5 * 24
    sun_end = 7 * 24
    ax.axvspan(sat_start, sun_end, alpha=0.12)

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if save:
        if filename is None:
            filename = f"{metric_col}_by_hour_of_week.png"
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_demand_hour_of_week_multi(
    df: pd.DataFrame,
    realized_col: str = "request_count",
    est_col: str | None = None,
    est_label: str = "Model estimate",
    time_col: str = DEFAULT_TIME_COL,
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Hour-of-week (0–167) demand pattern:

    - Realized demand: per-hour *average* of `realized_col`
    - Optional estimated demand: per-hour average of `est_col`
    - Baseline: pure hour-of-day pattern (0–23) averaged over full sample
    """
    if time_col not in df.columns or realized_col not in df.columns:
        print(f"Missing {time_col} or {realized_col}; skipping demand plot.")
        return
    if est_col is not None and est_col not in df.columns:
        print(f"Missing est_col '{est_col}'; disabling model overlay.")
        est_col = None

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    # Realized per hour-of-week (mean, not sum)
    g_real = _aggregate_by_hour_of_week(
        df=df,
        value_col=realized_col,
        time_col=time_col,
        agg="mean",
        curfew=curfew,
    ).rename(columns={"value": "realized"})

    # Optional model per hour-of-week
    if est_col is not None:
        g_est = _aggregate_by_hour_of_week(
            df=df,
            value_col=est_col,
            time_col=time_col,
            agg="mean",
            curfew=curfew,
        ).rename(columns={"value": "est"})
        g_real["est"] = g_est["est"]
    else:
        g_real["est"] = np.nan

    # Hour-of-day baseline (mean across full sample)
    d = df.copy()
    d["hour"] = d[time_col].dt.hour
    if curfew:
        if "request_hour" in d.columns:
            d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()
        else:
            d = d[~d["hour"].isin(CURFEW_HOURS)].copy()
    hourly_baseline = (
        d.groupby("hour")[realized_col]
        .mean()
        .reindex(range(24), fill_value=np.nan)
        .values
    )
    baseline_pattern = np.tile(hourly_baseline, 7)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Realized
    ax.plot(
        g_real["hour_of_week"],
        g_real["realized"],
        color="tab:gray",
        linewidth=1.4,
        label="Realized demand",
    )

    # Optional model
    if g_real["est"].notna().any():
        ax.plot(
            g_real["hour_of_week"],
            g_real["est"],
            color="tab:red",
            linestyle="--",
            linewidth=1.4,
            label=est_label,
        )

    # Baseline
    ax.plot(
        g_real["hour_of_week"],
        baseline_pattern,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Hour-of-day baseline",
    )

    # Day shading / ticks as in generic metric plot
    sat_start = 5 * 24
    sun_end = 7 * 24
    ax.axvspan(sat_start, sun_end, alpha=0.12)

    xticks = [24 * d + 12 for d in range(7)]
    xtick_labels = [calendar.day_name[d] for d in range(7)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    title_suffix = " (curfew 2–6 AM removed)" if curfew else ""
    ax.set_title(f"Demand by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0=Mon 00:00)")
    ax.set_ylabel(realized_col)
    ax.grid(True, axis="y", alpha=0.3)

    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()

    if save:
        fname = f"demand_by_hour_of_week_{realized_col}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_total_base_fare_by_hour_of_week(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Backward-compatible alias for `plot_base_fare_by_hour_of_week`.

    Historically, this helper had its own implementation; now it simply
    delegates to the main wrapper so that all base-fare hour-of-week plots
    are driven by the same generic core.
    """
    plot_base_fare_by_hour_of_week(
        df=df,
        time_col=DEFAULT_TIME_COL,
        fare_col="base_passenger_fare_sum",
        save=save,
        out_dir=out_dir,
        curfew=curfew,
    )


def plot_total_driver_pay_by_hour_of_week(df, save=False, out_dir=None, curfew=False):
    plot_metric_by_hour_of_week(
        df=df,
        metric_col="driver_pay_sum",
        agg="mean",
        save=save,
        out_dir=out_dir,
        curfew=curfew,
        ylabel="Avg total driver pay ($)",
        title="Avg total driver pay by hour of week",
        filename="total_driver_pay_by_hour_of_week.png",
        line_color="tab:red",
    )


def plot_margin_by_hour_of_week(df, save=False, out_dir=None, curfew=False):
    d = df.copy()
    d["margin_sum"] = d["base_passenger_fare_sum"] - d["driver_pay_sum"]

    plot_metric_by_hour_of_week(
        df=d,
        metric_col="margin_sum",
        agg="mean",
        save=save,
        out_dir=out_dir,
        curfew=curfew,
        ylabel="Avg total margin ($)",
        title="Avg total margin by hour of week",
        filename="margin_by_hour_of_week.png",
        line_color="black",
    )


def plot_base_fare_by_hour_of_week(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    fare_col: str = "base_passenger_fare_sum",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Total base fare by hour of week (0–167).

    This is now a thin wrapper around the generic hour-of-week metric core
    (`plot_metric_by_hour_of_week`) so that all hour-of-week charts share
    the same aggregation and plotting logic.

    Parameters
    ----------
    df : DataFrame
        Hourly-level metrics including `time_col` and `fare_col`.
    time_col : str, default DEFAULT_TIME_COL
        Datetime column at hourly resolution.
    fare_col : str, default "base_passenger_fare_sum"
        Column containing total base passenger fare per hour.
    save : bool, default False
        If True, write PNG to `out_dir` (or default plots dir).
    out_dir : Path or None, default None
        Output directory when `save=True`.
    curfew : bool, default False
        If True, hours in CURFEW_HOURS are dropped inside the core helper.
    """
    if time_col not in df.columns or fare_col not in df.columns:
        print("Missing datetime or fare column; skipping base-fare-by-hour-of-week plot.")
        return

    # Keep backward compatibility if caller passes a non-default time_col
    if time_col != DEFAULT_TIME_COL:
        df = df.copy()
        df[DEFAULT_TIME_COL] = df[time_col]

    plot_metric_by_hour_of_week(
        df=df,
        metric_col=fare_col,
        agg="mean",
        time_col=DEFAULT_TIME_COL,
        save=save,
        out_dir=out_dir,
        curfew=curfew,
        ylabel="Avg total base fare ($)",
        title="Avg total base fare by hour of week",
        filename="total_base_fare_by_hour_of_week.png",
        line_color="tab:green",
    )


def plot_fare_rate_vs_trips_hour_of_week(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    fare_sum_col: str = "base_passenger_fare_sum",
    miles_sum_col: str = "trip_miles_sum",
    time_sum_col: str = "trip_time_sum",      # NEW: total trip time (seconds)
    count_col: str = "request_count",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Hour-of-week (0–167) time series for:
      - avg base fare per trip   = total_base_fare / total_trips
      - avg base fare per mile   = total_base_fare / total_miles
      - avg base fare per minute = total_base_fare / (total_trip_time / 60)

    Helps see whether higher prices are driven by distance, time, or per-trip
    margins. All series are on one axis for now.
    """
    needed = {time_col, fare_sum_col, miles_sum_col, time_sum_col, count_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(
            "Skipping fare-rate-vs-trips hour-of-week plot; "
            f"missing columns: {missing}"
        )
        return

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek
    d["hour"] = d[time_col].dt.hour
    d["hour_of_week"] = d["dow"] * 24 + d["hour"]

    if curfew and "request_hour" in d.columns:
        d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()

    # Aggregate sums by hour-of-week
    agg = (
        d.groupby("hour_of_week", as_index=False)
        .agg(
            total_trips=(count_col, "sum"),
            total_base_fare=(fare_sum_col, "sum"),
            total_miles=(miles_sum_col, "sum"),
            total_time_sec=(time_sum_col, "sum"),
        )
        .sort_values("hour_of_week")
    )

    # Ensure all 0..167 present
    all_hours = pd.DataFrame({"hour_of_week": np.arange(168)})
    agg = all_hours.merge(agg, on="hour_of_week", how="left")

    # Compute rates
    total_trips = agg["total_trips"].replace({0: np.nan})
    total_miles = agg["total_miles"].replace({0: np.nan})
    total_time_min = (agg["total_time_sec"] / 60.0).replace({0: np.nan})

    fare_per_trip = agg["total_base_fare"] / total_trips
    fare_per_mile = agg["total_base_fare"] / total_miles
    fare_per_min = agg["total_base_fare"] / total_time_min

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg["hour_of_week"], fare_per_trip, label="Base fare per trip")
    ax.plot(agg["hour_of_week"], fare_per_mile, label="Base fare per mile")
    ax.plot(agg["hour_of_week"], fare_per_min, label="Base fare per minute")

    ax.set_title("Base fare rates vs hour of week")
    ax.set_xlabel("Hour of week (0=Mon 00:00)")
    ax.set_ylabel("Dollars")

    # Weekend shading
    ax.axvspan(113, 173, color="lightgray", alpha=0.2, label="Weekend (approx)")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)
        fig.savefig(out_dir / "fare_rate_vs_trips_hour_of_week.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Day-vs-weekday generic core and wrapper
# ---------------------------------------------------------------------------


def plot_metric_day_vs_weekday_avg(
    df: pd.DataFrame,
    target_date,
    metric_col: str,
    compare_dow1,
    compare_dow2=None,
    est_col: str | None = None,
    est_label: str = "Model estimate",
    time_col: str = DEFAULT_TIME_COL,
    hour_col: str = "request_hour",
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Generic version of the day-vs-weekday-average plot.

    Plots the hourly (0–23) values of `metric_col` for a specific calendar date
    against the average profile for one or two comparison weekdays.
    Optionally overlays a model-estimated series (est_col) for the target day.

    Examples:
        - Demand:
            plot_metric_day_vs_weekday_avg(
                df, "2025-01-19", "request_count",
                compare_dow1="Sunday", compare_dow2="Sunday",
                est_col="demand_est"
            )
        - Fare:
            plot_metric_day_vs_weekday_avg(
                df, "2025-01-19", "avg_base_passenger_fare",
                compare_dow1="Sunday", compare_dow2=None,
            )
    """
    # Normalise DOW spec
    dow1 = _normalize_dow(compare_dow1)
    dow2 = _normalize_dow(compare_dow2) if compare_dow2 is not None else None

    # Parse target date
    target_date = pd.to_datetime(target_date).date()

    # Basic column checks
    if time_col not in df.columns or hour_col not in df.columns:
        print("Missing time/hour columns; skipping day-vs-weekday plot.")
        return
    if metric_col not in df.columns:
        print(f"Missing metric column '{metric_col}'; skipping.")
        return
    if est_col is not None and est_col not in df.columns:
        print(f"Missing est_col '{est_col}'; disabling model overlay.")
        est_col = None

    # Output directory
    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    d = df.copy()
    d["date_only"] = d[time_col].dt.date
    d["dow"] = d[time_col].dt.dayofweek

    # Filter for target date’s rows
    day_rows = d[d["date_only"] == target_date].copy()
    if day_rows.empty:
        print(f"No data for target_date={target_date}; skipping day-vs-weekday plot.")
        return

    # Build hourly series for the target date
    day_metric = (
        day_rows.groupby(hour_col, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "metric_target"})
    )

    if est_col is not None:
        day_est = (
            day_rows.groupby(hour_col, as_index=False)[est_col]
            .mean()
            .rename(columns={est_col: "metric_est"})
        )
        day_metric = day_metric.merge(day_est, on=hour_col, how="left")
    else:
        day_metric["metric_est"] = np.nan

    # Build weekday-average comparisons
    def _weekday_avg(dow_val: int, label: str) -> pd.DataFrame:
        sub = d[d["dow"] == dow_val]
        if sub.empty:
            print(f"No rows for {label}; weekday average will be NaN.")
        return (
            sub.groupby(hour_col, as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: f"metric_{label}"})
        )

    w1_label = calendar.day_name[dow1]
    w1 = _weekday_avg(dow1, "w1")
    w1 = w1.rename(columns={f"metric_w1": f"metric_{w1_label}"})

    if dow2 is not None:
        w2_label = calendar.day_name[dow2]
        w2 = _weekday_avg(dow2, "w2")
        w2 = w2.rename(columns={f"metric_w2": f"metric_{w2_label}"})
    else:
        w2_label = None
        w2 = None

    # Merge all onto a full 0–23 frame
    hours = pd.DataFrame({hour_col: np.arange(24)})
    merged = hours.merge(day_metric, on=hour_col, how="left")
    merged = merged.merge(w1, on=hour_col, how="left")
    if w2 is not None:
        merged = merged.merge(w2, on=hour_col, how="left")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Target day
    ax.plot(
        merged[hour_col],
        merged["metric_target"],
        marker="o",
        label=f"{target_date} (actual)",
        linewidth=1.5,
    )

    # Optional model
    if merged["metric_est"].notna().any():
        ax.plot(
            merged[hour_col],
            merged["metric_est"],
            marker="o",
            linestyle="--",
            label=f"{target_date} ({est_label})",
            linewidth=1.5,
        )

    # Weekday averages
    ax.plot(
        merged[hour_col],
        merged[f"metric_{w1_label}"],
        marker="s",
        linestyle="-.",
        label=f"Avg {w1_label}",
        linewidth=1.2,
    )

    if w2_label is not None and f"metric_{w2_label}" in merged.columns:
        ax.plot(
            merged[hour_col],
            merged[f"metric_{w2_label}"],
            marker="^",
            linestyle=":",
            label=f"Avg {w2_label}",
            linewidth=1.2,
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(metric_col)
    ax.set_xticks(range(24))

    ttl_metric = metric_col.replace("_", " ")
    ax.set_title(
        f"{ttl_metric} — {target_date} vs weekday average(s)"
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()

    if save:
        date_str = pd.to_datetime(target_date).strftime("%Y%m%d")
        fname = f"day_vs_weekday_avg_{metric_col}_{date_str}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_day_vs_weekday_avg(
    df: pd.DataFrame,
    target_date,
    compare_dow1,
    compare_dow2=None,
    time_col: str = DEFAULT_TIME_COL,
    hour_col: str = "request_hour",
    count_col: str = "request_count",
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Thin wrapper over the generic day-vs-weekday core for demand.

    - metric_col        → count_col
    - no model line here (est_col=None)
    """
    plot_metric_day_vs_weekday_avg(
        df=df,
        target_date=target_date,
        metric_col=count_col,
        compare_dow1=compare_dow1,
        compare_dow2=compare_dow2,
        est_col=None,           # no model line in this wrapper
        est_label="Model estimate",
        time_col=time_col,
        hour_col=hour_col,
        save=save,
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Weather vs demand / price diagnostics
# ---------------------------------------------------------------------------


def plot_trips_vs_temperature(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Core diagnostic: trips vs various temperature measures."""
    _scatter_xy(
        df,
        x_col="temp_f_mean",
        y_col="request_count",
        title="Trips vs mean temperature",
        xlabel="Mean temperature (°F)",
        ylabel="Hourly request count",
        filename="core_trips_vs_temp_f_mean.png",
        save=save,
        out_dir=out_dir,
    )
    _scatter_xy(
        df,
        x_col="dewpoint_f_mean",
        y_col="request_count",
        title="Trips vs mean dewpoint",
        xlabel="Mean dewpoint (°F)",
        ylabel="Hourly request count",
        filename="core_trips_vs_dewpoint_f_mean.png",
        save=save,
        out_dir=out_dir,
    )
    _scatter_xy(
        df,
        x_col="heat_index_f",
        y_col="request_count",
        title="Trips vs heat index",
        xlabel="Heat index (°F)",
        ylabel="Hourly request count",
        filename="core_trips_vs_heat_index_f.png",
        save=save,
        out_dir=out_dir,
    )


def plot_trips_vs_wind_chill(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    _scatter_xy(
        df,
        x_col="wind_chill_f",
        y_col="request_count",
        title="Trips vs wind chill",
        xlabel="Wind chill (°F)",
        ylabel="Hourly request count",
        filename="core_trips_vs_wind_chill_f.png",
        save=save,
        out_dir=out_dir,
    )


def plot_trips_vs_precipitation(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    _scatter_xy(
        df,
        x_col="precip_1h_mm_total",
        y_col="request_count",
        title="Trips vs 1h precipitation",
        xlabel="Precipitation (mm)",
        ylabel="Hourly request count",
        filename="core_trips_vs_precip_1h_mm_total.png",
        save=save,
        out_dir=out_dir,
    )


def plot_price_metrics_vs_weather(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Price metrics (base fare, rider total, driver pay) vs key weather variables.
    """
    pairs = [
        ("temp_f_mean", "avg_base_passenger_fare"),
        ("temp_f_mean", "avg_rider_total"),
        ("temp_f_mean", "avg_driver_pay"),

        ("wind_chill_f", "avg_base_passenger_fare"),
        ("wind_chill_f", "avg_rider_total"),
        ("wind_chill_f", "avg_driver_pay"),

        ("heat_index_f", "avg_base_passenger_fare"),
        ("heat_index_f", "avg_rider_total"),
        ("heat_index_f", "avg_driver_pay"),

        ("precip_1h_mm_total", "avg_base_passenger_fare"),
        ("precip_1h_mm_total", "avg_rider_total"),
        ("precip_1h_mm_total", "avg_driver_pay"),
    ]

    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        fname = f"core_price_{y_col}_vs_{x_col}.png"
        _scatter_xy(
            df,
            x_col=x_col,
            y_col=y_col,
            title=f"{y_col} vs {x_col}",
            xlabel=x_col,
            ylabel=y_col,
            filename=fname,
            save=save,
            out_dir=out_dir,
        )


def plot_driver_pay_vs_fare_vs_weather(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Driver pay as % of base fare vs weather, plus simple scatter of driver pay vs base fare.
    """
    # Pay as % of base fare vs temp/wind_chill/heat_index
    pct_col = "driver_pay_pct_of_base_fare"
    weather_cols = ["temp_f_mean", "wind_chill_f", "heat_index_f", "precip_1h_mm_total"]

    for w in weather_cols:
        if w not in df.columns or pct_col not in df.columns:
            continue
        fname = f"core_driver_pay_pct_vs_{w}.png"
        _scatter_xy(
            df,
            x_col=w,
            y_col=pct_col,
            title=f"Driver pay % of base fare vs {w}",
            xlabel=w,
            ylabel=pct_col,
            filename=fname,
            save=save,
            out_dir=out_dir,
        )

    # Simple scatter: driver pay vs base fare
    if (
        "avg_base_passenger_fare" in df.columns
        and "avg_driver_pay" in df.columns
    ):
        _scatter_xy(
            df,
            x_col="avg_base_passenger_fare",
            y_col="avg_driver_pay",
            title="Avg driver pay vs avg base passenger fare",
            xlabel="Avg base fare",
            ylabel="Avg driver pay",
            filename="core_driver_pay_vs_base_fare.png",
            save=save,
            out_dir=out_dir,
        )


def plot_core_diagnostics(
    df: pd.DataFrame,
    save_plots: bool = False,
    out_dir: Path | None = None,
    curfew: bool = True,
) -> None:
    """
    Minimal diagnostic suite focusing on a few high-value plots:

      - demand hour-of-week (multi-series)
      - base fare / driver pay / margin hour-of-week
      - demand vs temperature-related metrics
      - simple precip & wait-time histograms
    """
    # Core demand hour-of-week
    plot_demand_hour_of_week_multi(
        df,
        realized_col="request_count",
        est_col=None,  # no model line by default here
        est_label="Model estimate",
        time_col=DEFAULT_TIME_COL,
        save=save_plots,
        out_dir=out_dir,
        curfew=curfew,
    )

    # Economics hour-of-week via generic core
    plot_total_base_fare_by_hour_of_week(
        df,
        save=save_plots,
        out_dir=out_dir,
        curfew=curfew,
    )
    plot_total_driver_pay_by_hour_of_week(
        df,
        save=save_plots,
        out_dir=out_dir,
        curfew=curfew,
    )
    plot_margin_by_hour_of_week(
        df,
        save=save_plots,
        out_dir=out_dir,
        curfew=curfew,
    )

    # Weather/demand & wait-time histograms
    plot_trips_vs_temperature(df, save=save_plots, out_dir=out_dir)
    plot_trips_vs_wind_chill(df, save=save_plots, out_dir=out_dir)
    plot_trips_vs_precipitation(df, save=save_plots, out_dir=out_dir)
    plot_weather_and_waittime_histograms(df, save=save_plots, out_dir=out_dir)


# ---------------------------------------------------------------------------
# Orchestrators (EDA run + day-vs-weekday CLI)
# ---------------------------------------------------------------------------


def run_eda(
    input_path: Path,
    save_plots: bool = False,
    out_dir: Path | None = None,
    curfew: bool = True,
    core: bool = False,
) -> None:
    df = _read_input(input_path)

    # Optional curfew filter: drop hours 2–6 based on request_hour
    if curfew and "request_hour" in df.columns:
        df = df[~df["request_hour"].isin(CURFEW_HOURS)].copy()

    # Basic sort by time if it exists
    if DEFAULT_TIME_COL in df.columns:
        df = df.sort_values(DEFAULT_TIME_COL)

    if core:
        # Core diagnostics only
        plot_core_diagnostics(df, save_plots=save_plots, out_dir=out_dir, curfew=curfew)
        return

    # Full EDA (existing behavior)
    plot_time_series(df, save=save_plots, out_dir=out_dir)
    plot_scatter_pairs(df, save=save_plots, out_dir=out_dir)
    plot_corr_heatmap(df, save=save_plots, out_dir=out_dir)

    plot_requests_by_hour_of_day(df, save=save_plots, out_dir=out_dir, curfew=curfew)
    plot_precip_distributions(df, save=save_plots, out_dir=out_dir)
    plot_weather_and_waittime_histograms(df, save=save_plots, out_dir=out_dir)
    plot_requests_by_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)
    plot_base_fare_by_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)
    plot_fare_rate_vs_trips_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)


def run_day_vs_weekday_from_file(
    file_name: str,
    target_date,
    compare_dow1,
    compare_dow2=None,
    curfew: bool = True,
    save_plots: bool = False,
) -> None:
    """
    Convenience wrapper:
      - Reads processed file
      - Applies optional curfew filter (drop hours 2–6)
      - Runs ONLY the day-vs-weekday comparison
    """
    input_path = PROCESSED_DIR / "tlc" / file_name
    df = _read_input(input_path)

    if curfew and "request_hour" in df.columns:
        df = df[~df["request_hour"].isin(CURFEW_HOURS)].copy()

    if DEFAULT_TIME_COL in df.columns:
        df = df.sort_values(DEFAULT_TIME_COL)

    plot_day_vs_weekday_avg(
        df,
        target_date=target_date,
        compare_dow1=compare_dow1,
        compare_dow2=compare_dow2,
        save=save_plots,
        out_dir=None,
    )
    print(f"Day vs weekday comparison complete for {target_date} from: {input_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA plotting for TLC + weather data.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to processed parquet (relative to project root or absolute).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="If set, write PNGs under plots/ instead of showing interactively.",
    )
    parser.add_argument(
        "--no-curfew",
        action="store_true",
        help="If set, do NOT drop curfew hours (2–6) from analyses.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="If set, run only the core diagnostics (faster, minimal plots).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Spyder-friendly main entry point
# ---------------------------------------------------------------------------

def main(
    file_name: str,
    save_plots: bool = False,
    curfew: bool = True,
    core: bool = False,
):
    """
    Run EDA from Spyder using F5.

    Example:
        main("fhvhv_lga_hourly_with_weather_2025.parquet",
             save_plots=True,
             curfew=True,
             core=True)
    """
    input_path = PROCESSED_DIR / file_name
    run_eda(
        input_path=input_path,
        save_plots=save_plots,
        out_dir=None,
        curfew=curfew,
        core=core,
    )
    print(f"EDA complete for: {input_path}")


if __name__ == "__main__":
    FILE_NAME = "fhvhv_lga_hourly_with_weather_2025.parquet"
    SAVE_PLOTS = False
    CURFEW = False     # True => drop 2–6AM
    CORE = False       # True => core-diagnostics-only

    main(
        FILE_NAME,
        save_plots=SAVE_PLOTS,
        curfew=CURFEW,
        core=CORE,
    )

