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

    ("avg_speed_mph", "request_count"),
    ("wait_time_sec_mean", "request_count"),

    ("avg_rider_total", "request_count"),
    ("avg_rider_total", "avg_driver_pay"),
    ("driver_pay_pct_of_base_fare", "avg_rider_total"),
    
    ("wind_chill_f", "avg_base_passenger_fare"),
    ("wind_chill_f", "avg_driver_pay"),
    ("heat_index_f", "avg_base_passenger_fare"),
    ("heat_index_f", "avg_driver_pay"),
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
# Helpers
# ---------------------------------------------------------------------------

def _ensure_out_dir(out_dir: Path | str) -> Path:
    """
    Ensure output directory exists; accept either Path or string.
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return df


def _make_precip_buckets(s: pd.Series) -> pd.Series:
    """Return categorical precip buckets: [0], (0,1], (1,2], ..., (49,50], (50,+]."""
    s = s.fillna(0).clip(lower=0)

    zero_mask = (s == 0)
    pos = s[~zero_mask]

    # bins: (0,1], (1,2], ..., (49,50], (50, inf]
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

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x_col], df[y_col], alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _normalize_dow(dow_spec) -> int:
    """
    Normalize a day-of-week spec to 0–6 (Monday=0).
    Accepts:
      - int 0–6
      - full name: 'Monday'
      - short name: 'Mon', 'mon', etc.
    """
    if isinstance(dow_spec, int):
        if 0 <= dow_spec <= 6:
            return dow_spec
        raise ValueError(f"Day-of-week int out of range 0–6: {dow_spec}")

    s = str(dow_spec).strip().lower()
    # Try full names
    for i in range(7):
        if s == calendar.day_name[i].lower():
            return i
    # Try 3-letter abbreviations
    for i in range(7):
        if s == calendar.day_name[i][:3].lower():
            return i

    raise ValueError(f"Unrecognized day-of-week spec: {dow_spec!r}")



# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    value_cols=None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Line plots of each metric vs time, with 24-hour moving average overlay.
    """
    if value_cols is None:
        value_cols = DEFAULT_TS_COLS

    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not in dataframe")

    # Keep only columns that exist
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols:
        print("No time series columns found in dataframe; skipping time series plots.")
        return

    # Ensure output directory exists if saving
    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    # Sort by time to ensure correct rolling behavior
    df_sorted = df.sort_values(time_col)

    for col in value_cols:
        # Convert to float for rolling ops (avoid dtype issues)
        y = df_sorted[col].astype(float)

        # 24-hour moving average (hourly frequency assumed)
        y_ma = y.rolling(window=24, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5.5))

        # Raw series
        ax.plot(
            df_sorted[time_col],
            y,
            linewidth=0.7,
            alpha=0.45,
            label=f"{col} (hourly)",
        )

        # 24-hour moving average in dark red
        ax.plot(
            df_sorted[time_col],
            y_ma,
            linewidth=1.6,
            color="darkred",
            label=f"{col} (24h MA)",
        )

        ax.set_title(f"{col} over time")
        ax.set_xlabel(time_col)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.autofmt_xdate()

        # Save or show
        if save:
            fname = f"time_{col}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)


def plot_scatter_pairs(
    df: pd.DataFrame,
    pairs=None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Scatter plots for selected x/y pairs to eyeball correlations."""
    if pairs is None:
        pairs = DEFAULT_SCATTER_PAIRS

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skipping scatter {x_col} vs {y_col}: one or both missing from dataframe.")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df[x_col], df[y_col], alpha=0.4)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

        if save:
            fname = f"scatter_{y_col}_vs_{x_col}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)


def plot_corr_heatmap(
    df: pd.DataFrame,
    cols=None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Correlation matrix heatmap for selected numeric columns."""
    if cols is None:
        cols = DEFAULT_CORR_COLS

    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        print("Fewer than 2 correlation columns found; skipping correlation heatmap.")
        return

    corr = df[cols].corr()

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    fig, ax = plt.subplots(figsize=(0.8 * len(cols) + 3, 0.8 * len(cols) + 3))
    im = ax.imshow(corr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    # Annotate correlation values
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save:
        fname = "corr_matrix.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_requests_by_hour_of_day(
    df: pd.DataFrame,
    count_col: str = "request_count",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Total requests by hour of day.

    If curfew=False: simple 0–23 bar chart.
    If curfew=True: hours 2–6 removed, shown with a broken x-axis (0–1 | 7–23).
    """
    if "request_hour" not in df.columns or count_col not in df.columns:
        print("Missing request_hour or request_count; skipping requests-by-hour-of-day plot.")
        return

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    grouped = df.groupby("request_hour")[count_col].sum()

    # ---------------- no-curfew: simple bar chart -----------------
    if not curfew:
        grouped_full = grouped.reindex(range(24), fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(grouped_full.index, grouped_full.values)
        ax.set_title("Total requests by hour of day")
        ax.set_xlabel("Hour of day (0–23)")
        ax.set_ylabel("Total requests")
        ax.set_xticks(range(24))
        ax.grid(True, axis="y", alpha=0.3)

        if save:
            fig.savefig(out_dir / "hist_requests_by_hour_of_day.png",
                        dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
        return

    # ---------------- curfew=True: broken x-axis -------------------
    valid_hours_left = [0, 1]
    valid_hours_right = list(range(7, 24))

    left_vals = grouped.reindex(valid_hours_left, fill_value=0)
    right_vals = grouped.reindex(valid_hours_right, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 5]},
    )

    # Left: 0–1
    ax1.bar(left_vals.index, left_vals.values)
    ax1.set_xticks(valid_hours_left)
    ax1.set_xlim(-0.5, 1.5)

    # Right: 7–23
    ax2.bar(right_vals.index, right_vals.values)
    ax2.set_xticks(valid_hours_right)
    ax2.set_xlim(6.5, 23.5)

    # Title / labels
    fig.suptitle("Total requests by hour of day (hours 2–6 removed)")
    ax1.set_ylabel("Total requests")
    ax2.set_xlabel("Hour of day")

    # --- Clean up spines / ticks so inner vertical axes disappear ---
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False)  # hide y tick labels in right panel

    # Grid only once (looks cleaner)
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.grid(False)

    # --- Diagonal break marks (same size on both sides) ---
    d = 0.02  # size of the diagonal lines in axes coordinates
    # right side of left axis
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    # left side of right axis
    kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        fig.savefig(out_dir / "hist_requests_by_hour_of_day_broken.png",
                    dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_precip_distributions(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
    count_col: str = "request_count",
) -> None:
    if PRECIP_COL not in df.columns:
        print("No precip column found; skipping precip distribution plots.")
        return

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    # Build buckets
    buckets = _make_precip_buckets(df[PRECIP_COL])

    # === REMOVE ZERO BUCKET ====================================================
    nonzero_mask = buckets != "0"
    buckets_nz = buckets[nonzero_mask]

    df_nz = df.loc[nonzero_mask].copy()
    df_nz["precip_bucket"] = buckets_nz

    # === HOURS BY PRECIP BUCKET (non-zero) ==================================
    hours_by_bucket = buckets_nz.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hours_by_bucket.index.astype(str), hours_by_bucket.values)
    ax.set_title("Hours by precipitation bucket (non-zero only)")
    ax.set_xlabel("Precipitation bucket (mm)")
    ax.set_ylabel("Number of hours")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    # numeric median precip (mm)
    precip_nz = df[PRECIP_COL]
    precip_nz = precip_nz[precip_nz > 0]
    median_precip_hours = precip_nz.median()

    ax.axvline(
        x=hours_by_bucket.index[(hours_by_bucket.cumsum() >= hours_by_bucket.sum()/2)].tolist()[0],
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Median precip ≈ {median_precip_hours:.2f} mm",
    )
    ax.legend()


    if save:
        fig.savefig(out_dir / "hist_hours_by_precip_bucket_nonzero.png",
                    dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    # === REQUESTS BY PRECIP BUCKET (non-zero) ===============================
    requests_by_bucket = (
        df_nz.groupby("precip_bucket")[count_col]
        .sum()
        .reindex(hours_by_bucket.index, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(requests_by_bucket.index.astype(str), requests_by_bucket.values)
    ax.set_title("Total requests by precipitation bucket (non-zero only)")
    ax.set_xlabel("Precipitation bucket (mm)")
    ax.set_ylabel("Total requests")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    precip_weighted = np.repeat(
        precip_nz.values,
        repeats=1  # each row is one hour; counts are already in y-axis
    )
    median_precip_requests = np.median(precip_weighted) if precip_weighted.size > 0 else np.nan

    ax.axvline(
        x=requests_by_bucket.index[(requests_by_bucket.cumsum() >= requests_by_bucket.sum()/2)].tolist()[0],
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Median precip ≈ {median_precip_requests:.2f} mm",
    )
    ax.legend()

    if save:
        fig.savefig(out_dir / "requests_by_precip_bucket_nonzero.png",
                    dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_weather_and_waittime_histograms(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    if out_dir is None and save:
        out_dir = PLOTS_DIR

    # Wind chill / heat index (°F), 5-degree buckets
    if "wind_chill_f" in df.columns:
        _plot_simple_hist(
            df["wind_chill_f"],
            "Distribution of hourly wind chill",
            "Wind chill (°F)",
            "hist_wind_chill_f.png",
            save,
            out_dir,
            bin_width=5.0,
        )

    if "heat_index_f" in df.columns:
        _plot_simple_hist(
            df["heat_index_f"],
            "Distribution of hourly heat index",
            "Heat index (°F)",
            "hist_heat_index_f.png",
            save,
            out_dir,
            bin_width=5.0,
        )

    # Wait times (seconds) – overall
    for col in ["wait_time_sec_mean", "wait_time_sec_median"]:
        if col in df.columns:
            _plot_simple_hist(
                df[col],
                f"Distribution of {col}",
                "Wait time (seconds)",
                f"hist_{col}.png",
                save,
                out_dir,
                bin_width=30.0,  # 30-second buckets; adjust if needed
            )

    # Wait times conditional on precip > 0
    if PRECIP_COL in df.columns:
        wet = df[df[PRECIP_COL] > 0]
        if not wet.empty:
            for col in ["wait_time_sec_mean", "wait_time_sec_median"]:
                if col in wet.columns:
                    _plot_simple_hist(
                        wet[col],
                        f"Distribution of {col} (precip > 0)",
                        "Wait time (seconds)",
                        f"hist_{col}_precip_gt0.png",
                        save,
                        out_dir,
                        bin_width=30.0,
                    )


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


def plot_metric_day_vs_weekday_avg(
    df: pd.DataFrame,
    target_date,
    metric_col: str,
    compare_dow1,
    compare_dow2=None,
    est_col: str | None = None,
    est_label: str = "Model estimate",
    shade_hours: tuple[int, int] | None = None,
    time_col: str = DEFAULT_TIME_COL,
    hour_col: str = "request_hour",
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Generic version of the day-vs-weekday-average plot.

    Plots the hourly (0–23) values of `metric_col` for a specific calendar date
    against the average profile for one or two comparison weekdays.
    Optionally overlays a model-estimated series (est_col) for the target day
    and an optional shaded hour range.
    """
    # ------------------------------------------------------------------
    # Basic column checks
    # ------------------------------------------------------------------
    if time_col not in df.columns or hour_col not in df.columns:
        print("Missing time/hour columns; skipping day-vs-weekday plot.")
        return
    if metric_col not in df.columns:
        print(f"Missing metric column '{metric_col}'; skipping.")
        return
    if est_col is not None and est_col not in df.columns:
        print(f"Missing est_col '{est_col}'; disabling model overlay.")
        est_col = None

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    # ------------------------------------------------------------------
    # Normalize target date
    # ------------------------------------------------------------------
    target_ts = pd.to_datetime(target_date)
    target_date_only = target_ts.date()

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek          # 0 = Monday ... 6 = Sunday
    d["date_only"] = d[time_col].dt.date         # Python date

    # Mask for the target calendar date
    day_mask = d["date_only"] == target_date_only
    if not day_mask.any():
        print(f"No data found for date {target_date_only}; skipping.")
        return

    target_dow = target_ts.dayofweek
    target_dow_name = calendar.day_name[target_dow]

    # ------------------------------------------------------------------
    # Build the realized hourly profile (0–23)
    # ------------------------------------------------------------------
    day_profile = (
        d.loc[day_mask]
        .groupby(hour_col)[metric_col]
        .sum()
        .reindex(range(24), fill_value=0.0)
    )

    # Optional model-estimated hourly profile for that day
    est_profile = None
    if est_col is not None:
        est_profile = (
            d.loc[day_mask]
            .groupby(hour_col)[est_col]
            .sum()
            .reindex(range(24), fill_value=0.0)
        )

    # ------------------------------------------------------------------
    # Resolve weekday comparison(s)
    # ------------------------------------------------------------------
    def _normalize_dow(dow_spec) -> int:
        """Normalize weekday name/int to canonical 0–6."""
        if isinstance(dow_spec, int):
            if 0 <= dow_spec <= 6:
                return dow_spec
            raise ValueError(f"Invalid weekday int: {dow_spec}")

        s = str(dow_spec).strip().lower()
        for i in range(7):
            if s == calendar.day_name[i].lower():
                return i
        for i in range(7):
            if s == calendar.day_name[i][:3].lower():
                return i

        raise ValueError(f"Unrecognized DOW: {dow_spec}")

    dow1 = _normalize_dow(compare_dow1)
    dow1_name = calendar.day_name[dow1]

    dow2 = None
    dow2_name = None
    if compare_dow2 is not None:
        dow2 = _normalize_dow(compare_dow2)
        dow2_name = calendar.day_name[dow2]

    # ------------------------------------------------------------------
    # Average weekday profile(s)
    # ------------------------------------------------------------------
    def _avg_profile_for_dow(dow: int) -> pd.Series:
        mask = (d["dow"] == dow) & (~day_mask)
        if not mask.any():
            return pd.Series(0.0, index=range(24))
        return (
            d.loc[mask]
            .groupby(hour_col)[metric_col]
            .mean()
            .reindex(range(24), fill_value=0.0)
        )

    avg_profile_1 = _avg_profile_for_dow(dow1)
    avg_profile_2 = _avg_profile_for_dow(dow2) if dow2 is not None else None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Optional shaded region on the x-axis
    if shade_hours is not None:
        start, end = shade_hours
        x0, x1 = sorted((start, end))
        ax.axvspan(x0 - 0.5, x1 + 0.5, alpha=0.22, color="cornflowerblue")

    # Realized metric: solid gray + DOW-colored markers
    real_color = DAY_COLOR_MAP.get(target_dow, "tab:gray")

    ax.plot(
        hours,
        day_profile.values,
        color="0.4",
        linewidth=1.6,
        label=f"{target_date_only} ({target_dow_name})",
    )

    ax.scatter(
        hours,
        day_profile.values,
        color=real_color,
        s=25,
        alpha=0.9,
    )

    # Model estimate overlay (optional, DISTINCT style)
    if est_profile is not None:
        ax.plot(
            hours,
            est_profile.values,
            linestyle=":",        # dotted
            linewidth=1.6,
            color="tab:red",      # red; different from gray/black/blue
            label=est_label,
        )

    # Baseline weekday 1: black dashed
    ax.plot(
        hours,
        avg_profile_1.values,
        linestyle="--",
        linewidth=1.0,
        color="black",
        label=f"Avg {dow1_name}",
    )

    # Baseline weekday 2 (optional): blue dashed
    if avg_profile_2 is not None:
        ax.plot(
            hours,
            avg_profile_2.values,
            linestyle="--",
            linewidth=1.0,
            color="tab:blue",
            label=f"Avg {dow2_name}",
        )

    # Titles / labels
    ax.set_title(f"{metric_col} on {target_date_only} vs weekday averages")
    ax.set_xlabel("Hour of day (0–23)")
    ax.set_ylabel(metric_col)
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    # Save or show
    if save:
        date_str = target_ts.strftime("%Y%m%d")
        safe_metric = str(metric_col).replace(" ", "_")
        fname = f"day_vs_weekday_avg_{safe_metric}_{date_str}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)



def plot_base_fare_by_hour_of_week(
    df: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    fare_col: str = "base_passenger_fare_sum",
    save: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Total base fare by hour of week (0–167), with:
      - continuous line over 0–167
      - points color-coded by day of week
      - weekend shaded (Fri 17:00 -> Mon 05:00)
    Assumes any curfew filtering (dropping hours 2–6) was already
    applied upstream; `curfew` is only used for the plot title suffix.
    """
    if time_col not in df.columns or fare_col not in df.columns:
        print("Missing datetime or fare column; skipping base-fare-by-hour-of-week plot.")
        return

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek        # 0 = Monday ... 6 = Sunday
    d["hour"] = d[time_col].dt.hour            # 0–23
    d["hour_of_week"] = d["dow"] * 24 + d["hour"]  # 0–167

    # Aggregate TOTAL base fare per hour-of-week
    grouped = (
        d.groupby("hour_of_week", as_index=False)[fare_col]
        .mean()
        .rename(columns={fare_col: "avg_base_fare"})
        .sort_values("hour_of_week")
    )

    # Ensure all 0..167 present
    all_hours = pd.DataFrame({"hour_of_week": np.arange(168)})
    grouped = all_hours.merge(grouped, on="hour_of_week", how="left")
    grouped["avg_base_fare"] = grouped["avg_base_fare"].fillna(np.nan)

    # Day-of-week for each hour_of_week
    grouped["dow"] = grouped["hour_of_week"] // 24

    fig, ax = plt.subplots(figsize=(14, 5))

    # Continuous line
    ax.plot(
        grouped["hour_of_week"],
        grouped["avg_base_fare"],
        linewidth=1.0,
        color="dimgray",
    )

    # Color-coded points by DOW
    for dow in range(7):
        mask = grouped["dow"] == dow
        if not mask.any():
            continue
        ax.scatter(
            grouped.loc[mask, "hour_of_week"],
            grouped.loc[mask, "avg_base_fare"],
            s=15,
            alpha=0.8,
            label=calendar.day_name[dow],
        )

    title_suffix = " (hours 2–6 removed)" if curfew else ""
    ax.set_title(f"Average base passenger fare by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0–167)")
    ax.set_ylabel("Average base passenger fare ($)")
    ax.grid(True, alpha=0.3)

    # X-ticks at day boundaries
    xticks = [24 * d for d in range(8)]
    xtick_labels = [calendar.day_name[d] for d in range(7)] + ["Mon (next week)"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Shade weekend: Fri 17:00 -> Mon 05:00
    weekend_start = 4 * 24 + 19   # 113
    weekend_end1 = 7 * 24         # 168
    weekend_end2 = 6              # 5

    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=7,
        fontsize=8,
        frameon=False
    )
    fig.tight_layout()

    if save:
        fig.savefig(out_dir / "base_fare_by_hour_of_week.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


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

    Formatted like plot_requests_by_hour_of_week:
      - continuous line over 0–167
      - points color-coded by day of week
      - weekend shaded (Fri 19:00 -> Mon 06:00)
      - legend above chart, 7 columns

    Assumes any curfew filtering (dropping hours 2–6) was already
    applied upstream; `curfew` is only used for the title suffix.
    """
    # Basic checks
    if time_col not in df.columns:
        print("Missing datetime column; skipping fare-rate hour-of-week plots.")
        return
    for col in (fare_sum_col, miles_sum_col, time_sum_col, count_col):
        if col not in df.columns:
            print(f"Missing {col}; skipping fare-rate hour-of-week plots.")
            return

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek          # 0 = Monday ... 6 = Sunday
    d["hour"] = d[time_col].dt.hour              # 0–23
    d["hour_of_week"] = d["dow"] * 24 + d["hour"]  # 0–167

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

    # Derived metrics
    agg["fare_per_trip"] = np.where(
        agg["total_trips"] > 0,
        agg["total_base_fare"] / agg["total_trips"],
        np.nan,
    )
    agg["fare_per_mile"] = np.where(
        agg["total_miles"] > 0,
        agg["total_base_fare"] / agg["total_miles"],
        np.nan,
    )

    total_time_min = agg["total_time_sec"] / 60.0
    agg["fare_per_minute"] = np.where(
        total_time_min > 0,
        agg["total_base_fare"] / total_time_min,
        np.nan,
    )

    # Day-of-week for each hour_of_week (0..6)
    agg["dow"] = agg["hour_of_week"] // 24

    title_suffix = " (hours 2–6 removed)" if curfew else ""

    # Common x-axis ticks and weekend shading boundaries
    xticks = [24 * d for d in range(8)]
    xtick_labels = [calendar.day_name[d] for d in range(7)] + ["Mon (next week)"]

    weekend_start = 4 * 24 + 19   # Friday 19:00 = hour 115
    weekend_end1 = 7 * 24         # 168
    weekend_end2 = 6              # Monday 06:00 = hour 6

    # ------------------- 1) fare_per_trip time series -------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        agg["hour_of_week"],
        agg["fare_per_trip"],
        linewidth=1.0,
        color="dimgray",
    )

    for dow in range(7):
        mask = (agg["dow"] == dow) & agg["fare_per_trip"].notna()
        if not mask.any():
            continue
        ax.scatter(
            agg.loc[mask, "hour_of_week"],
            agg.loc[mask, "fare_per_trip"],
            s=15,
            alpha=0.8,
            label=calendar.day_name[dow],
        )

    ax.set_title(f"Avg base fare per trip by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0–167)")
    ax.set_ylabel("Avg base fare per trip ($)")
    ax.grid(True, alpha=0.3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=7,
        fontsize=8,
        frameon=False
    )

    fig.tight_layout()

    if save:
        fig.savefig(
            out_dir / "fare_per_trip_by_hour_of_week.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close(fig)

    # ------------------- 2) fare_per_mile time series -------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        agg["hour_of_week"],
        agg["fare_per_mile"],
        linewidth=1.0,
        color="dimgray",
    )

    for dow in range(7):
        mask = (agg["dow"] == dow) & agg["fare_per_mile"].notna()
        if not mask.any():
            continue
        ax.scatter(
            agg.loc[mask, "hour_of_week"],
            agg.loc[mask, "fare_per_mile"],
            s=15,
            alpha=0.8,
            label=calendar.day_name[dow],
        )

    ax.set_title(f"Avg base fare per mile by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0–167)")
    ax.set_ylabel("Avg base fare per mile ($/mile)")
    ax.grid(True, alpha=0.3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=7,
        fontsize=8,
        frameon=False
    )

    fig.tight_layout()

    if save:
        fig.savefig(
            out_dir / "fare_per_mile_by_hour_of_week.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close(fig)

    # ------------------- 3) fare_per_minute time series -------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        agg["hour_of_week"],
        agg["fare_per_minute"],
        linewidth=1.0,
        color="dimgray",
    )

    for dow in range(7):
        mask = (agg["dow"] == dow) & agg["fare_per_minute"].notna()
        if not mask.any():
            continue
        ax.scatter(
            agg.loc[mask, "hour_of_week"],
            agg.loc[mask, "fare_per_minute"],
            s=15,
            alpha=0.8,
            label=calendar.day_name[dow],
        )

    ax.set_title(f"Avg base fare per minute by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0–167)")
    ax.set_ylabel("Avg base fare per minute ($/min)")
    ax.grid(True, alpha=0.3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=7,
        fontsize=8,
        frameon=False
    )

    fig.tight_layout()

    if save:
        fig.savefig(
            out_dir / "fare_per_minute_by_hour_of_week.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close(fig)


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
    line_color: str = "dimgray",
) -> None:
    """
    Generic hour-of-week (0–167) plot for a single metric.

    - Aggregates `metric_col` by hour_of_week with the given `agg`.
    - Draws a continuous line (line_color) and day-colored points.
    - Shades weekend (Fri 19:00 -> Mon 06:00).
    - Legend is placed below the x-axis in 7 columns.

    This is intended to be reused for base fare, driver pay, margin, etc.
    """
    if time_col not in df.columns or metric_col not in df.columns:
        print(f"Missing {time_col} or {metric_col}; skipping hour-of-week plot.")
        return

    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek        # 0 = Monday ... 6 = Sunday
    d["hour"] = d[time_col].dt.hour            # 0–23
    d["hour_of_week"] = d["dow"] * 24 + d["hour"]  # 0–167

    # Optional curfew filter at this level (if not already applied upstream)
    if curfew and "request_hour" in d.columns:
        d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()

    # Aggregate metric by hour_of_week
    grouped = (
        d.groupby("hour_of_week", as_index=False)[metric_col]
        .agg(agg)
        .rename(columns={metric_col: "metric"})
        .sort_values("hour_of_week")
    )

    # Ensure all 0..167 present
    all_hours = pd.DataFrame({"hour_of_week": np.arange(168)})
    grouped = all_hours.merge(grouped, on="hour_of_week", how="left")
    grouped["metric"] = grouped["metric"].astype(float)

    # Day-of-week for each hour_of_week
    grouped["dow"] = grouped["hour_of_week"] // 24

    # X-ticks at day boundaries
    xticks = [24 * d for d in range(8)]
    xtick_labels = [calendar.day_name[d] for d in range(7)] + ["Mon (next week)"]

    # Weekend shading: Fri 19:00 -> Mon 06:00
    weekend_start = 4 * 24 + 19   # 115
    weekend_end1 = 7 * 24         # 168
    weekend_end2 = 6              # 6

    fig, ax = plt.subplots(figsize=(14, 5))

    # Continuous line
    ax.plot(
        grouped["hour_of_week"],
        grouped["metric"],
        linewidth=1.0,
        color=line_color,
        zorder=1,
    )

    # Color-coded points by DOW (same color order as before)
    for dow in range(7):
        mask = grouped["dow"] == dow
        if not mask.any():
            continue
        ax.scatter(
            grouped.loc[mask, "hour_of_week"],
            grouped.loc[mask, "metric"],
            s=15,
            alpha=0.8,
            label=calendar.day_name[dow],
            zorder=2,
        )

    if title is None:
        title = f"{metric_col} by hour of week"
    ax.set_title(title)

    if ylabel is None:
        ylabel = metric_col
    ax.set_ylabel(ylabel)

    ax.set_xlabel("Hour of week (0–167)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Weekend shading
    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    # Legend below x-axis, 7 columns (what you asked for)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=7,
        fontsize=8,
        frameon=False,
    )

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
    Hour-of-week (0–167) demand plot with:
      - Realized demand (aggregated by hour_of_week)
      - Optional model-estimated demand (same aggregation)
      - Black dashed hourly-of-day baseline pattern (avg across hours 0–23)

    Realized series uses the usual DOW-colored markers.
    Optional model series is a dark red dash-dot line.
    """
    # Basic checks
    if time_col not in df.columns:
        print(f"Missing {time_col}; skipping demand hour-of-week plot.")
        return
    if realized_col not in df.columns:
        print(f"Missing {realized_col}; skipping demand hour-of-week plot.")
        return
    if est_col is not None and est_col not in df.columns:
        print(f"Missing {est_col}; skipping model-estimate overlay.")
        est_col = None  # disable

    # Output directory
    if save:
        if out_dir is None:
            out_dir = PLOTS_DIR
        out_dir = _ensure_out_dir(out_dir)

    d = df.copy()
    d["dow"] = d[time_col].dt.dayofweek          # 0=Mon..6=Sun
    d["hour"] = d[time_col].dt.hour              # 0–23

    # Optional curfew filter (LaGuardia 2–6)
    if curfew and "request_hour" in d.columns:
        d = d[~d["request_hour"].isin(CURFEW_HOURS)].copy()

    d["hour_of_week"] = d["dow"] * 24 + d["hour"]  # 0–167

    # ------------------------------------------------------------------
    # 1) Realized demand by hour_of_week
    # ------------------------------------------------------------------
    g_real = (
        d.groupby("hour_of_week", as_index=False)[realized_col]
        .sum()
        .rename(columns={realized_col: "realized"})
        .sort_values("hour_of_week")
    )

    # Ensure full 0..167 grid
    all_hours = pd.DataFrame({"hour_of_week": np.arange(168)})
    g_real = all_hours.merge(g_real, on="hour_of_week", how="left")
    g_real["realized"] = g_real["realized"].fillna(0.0)

    # Day-of-week per hour_of_week
    g_real["dow"] = g_real["hour_of_week"] // 24

    # ------------------------------------------------------------------
    # 2) Optional model estimate by hour_of_week
    # ------------------------------------------------------------------
    if est_col is not None:
        g_est = (
            d.groupby("hour_of_week", as_index=False)[est_col]
            .sum()
            .rename(columns={est_col: "est"})
            .sort_values("hour_of_week")
        )
        g_est = all_hours.merge(g_est, on="hour_of_week", how="left")
        g_real["est"] = g_est["est"].astype(float)
    else:
        g_real["est"] = np.nan

    # ------------------------------------------------------------------
    # 3) Hour-of-day baseline pattern (0–23), repeated over the week
    # ------------------------------------------------------------------
    hourly_totals = (
        d.groupby("hour")[realized_col]
        .sum()
        .reindex(range(24), fill_value=0.0)
    )
    # As per your earlier spec: divide the bar chart by 7
    hourly_avg = hourly_totals / 7.0
    hourly_avg_pattern = np.tile(hourly_avg.values, 7)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    # Realized: continuous line + DOW-colored markers
    ax.plot(
        g_real["hour_of_week"],
        g_real["realized"],
        linewidth=1.0,
        color="dimgray",
        label="Realized demand",
        zorder=1,
    )

    for dow in range(7):
        mask = g_real["dow"] == dow
        if not mask.any():
            continue
        ax.scatter(
            g_real.loc[mask, "hour_of_week"],
            g_real.loc[mask, "realized"],
            s=15,
            alpha=0.8,
            color=DAY_COLOR_MAP.get(dow, "tab:gray"),
            label=calendar.day_name[dow],
            zorder=2,
        )

    # Baseline: black dashed hourly-of-day pattern
    ax.plot(
        g_real["hour_of_week"],
        hourly_avg_pattern,
        linestyle="--",
        linewidth=1.0,
        color="black",
        label="Hourly avg (0–23 pattern)",
        zorder=0,
    )

    # Optional model estimate: dark red dash-dot
    if est_col is not None and g_real["est"].notna().any():
        ax.plot(
            g_real["hour_of_week"],
            g_real["est"],
            linestyle="-.",
            linewidth=1.2,
            color="darkred",
            label=est_label,
            zorder=1.5,
        )

    title_suffix = " (hours 2–6 removed)" if curfew else ""
    ax.set_title(f"Demand by hour of week{title_suffix}")
    ax.set_xlabel("Hour of week (0–167)")
    ax.set_ylabel(realized_col)
    ax.grid(True, alpha=0.3)

    # X-ticks at day boundaries
    xticks = [24 * d for d in range(8)]
    xtick_labels = [calendar.day_name[d] for d in range(7)] + ["Mon (next week)"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Weekend shading: Fri 19:00 -> Mon 06:00
    weekend_start = 4 * 24 + 19   # 115
    weekend_end1 = 7 * 24         # 168
    weekend_end2 = 6              # 6
    ax.axvspan(weekend_start, weekend_end1 - 1, alpha=0.12)
    ax.axvspan(0, weekend_end2, alpha=0.12)

    # Legend below x axis (7 DOW colors + baseline + optional model)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),  # tuned a bit lower to avoid overlap
        ncol=7,
        fontsize=8,
        frameon=False,
    )

    # Give extra bottom margin so legend isn't clipped
    plt.subplots_adjust(bottom=0.22)
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
    Average total base passenger fare by hour of week (0–167).

    - Uses base_passenger_fare_sum at the hourly level.
    - Aggregates with mean → expected total hourly revenue for that hour-of-week.
    - Dark green line, day-colored points, weekend shading, legend below x-axis.
    """
    plot_metric_by_hour_of_week(
        df=df,
        metric_col="base_passenger_fare_sum",
        agg="mean",
        save=save,
        out_dir=out_dir,
        curfew=curfew,
        ylabel="Avg total base passenger fare ($)",
        title="Avg total base fare by hour of week",
        filename="total_base_fare_by_hour_of_week.png",
        line_color="darkgreen",  # money
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


def plot_day_vs_weekday_avg(
    df: pd.DataFrame,
    target_date,
    compare_dow1,
    compare_dow2=None,
    time_col: str = DEFAULT_TIME_COL,
    hour_col: str = "request_hour",
    count_col: str = "request_count",
    est_col: str | None = None,
    shade_hours: tuple[int, int] | None = None,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    plot_metric_day_vs_weekday_avg(
        df=df,
        target_date=target_date,
        metric_col=count_col,
        compare_dow1=compare_dow1,
        compare_dow2=compare_dow2,
        est_col=est_col,
        shade_hours=shade_hours,
        time_col=time_col,
        hour_col=hour_col,
        save=save,
        out_dir=out_dir,
    )




# ---------------------------------------------------------------------------
# Core Plotting Functions
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
        title="Trips vs 1-hour precipitation",
        xlabel="Precipitation (mm, 1-hour total)",
        ylabel="Hourly request count",
        filename="core_trips_vs_precip_1h_mm.png",
        save=save,
        out_dir=out_dir,
    )


def plot_price_metrics_vs_weather(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    # Rider total vs core weather
    _scatter_xy(
        df,
        x_col="temp_f_mean",
        y_col="avg_rider_total",
        title="Avg rider total vs mean temperature",
        xlabel="Mean temperature (°F)",
        ylabel="Avg rider total ($)",
        filename="core_price_vs_temp.png",
        save=save,
        out_dir=out_dir,
    )
    _scatter_xy(
        df,
        x_col="wind_chill_f",
        y_col="avg_rider_total",
        title="Avg rider total vs wind chill",
        xlabel="Wind chill (°F)",
        ylabel="Avg rider total ($)",
        filename="core_price_vs_wind_chill.png",
        save=save,
        out_dir=out_dir,
    )
    _scatter_xy(
        df,
        x_col="precip_1h_mm_total",
        y_col="avg_rider_total",
        title="Avg rider total vs precipitation",
        xlabel="Precipitation (mm, 1-hour total)",
        ylabel="Avg rider total ($)",
        filename="core_price_vs_precip.png",
        save=save,
        out_dir=out_dir,
    )


def plot_driver_pay_vs_fare_vs_weather(
    df: pd.DataFrame,
    save: bool = False,
    out_dir: Path | None = None,
) -> None:
    # Baseline: fare vs driver pay
    _scatter_xy(
        df,
        x_col="avg_rider_total",
        y_col="avg_driver_pay",
        title="Avg rider total vs avg driver pay",
        xlabel="Avg rider total ($)",
        ylabel="Avg driver pay ($)",
        filename="core_driverpay_vs_fare.png",
        save=save,
        out_dir=out_dir,
    )

    # Driver pay share vs weather
    _scatter_xy(
        df,
        x_col="temp_f_mean",
        y_col="driver_pay_pct_of_base_fare",
        title="Driver pay % of base fare vs temperature",
        xlabel="Mean temperature (°F)",
        ylabel="Driver pay % of base fare",
        filename="core_driverpay_pct_vs_temp.png",
        save=save,
        out_dir=out_dir,
    )
    _scatter_xy(
        df,
        x_col="precip_1h_mm_total",
        y_col="driver_pay_pct_of_base_fare",
        title="Driver pay % of base fare vs precipitation",
        xlabel="Precipitation (mm, 1-hour total)",
        ylabel="Driver pay % of base fare",
        filename="core_driverpay_pct_vs_precip.png",
        save=save,
        out_dir=out_dir,
    )



# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_core_diagnostics(
    df: pd.DataFrame,
    save_plots: bool = False,
    out_dir: Path | None = None,
    curfew: bool = False,
) -> None:
    """
    Run ONLY the core pre-model diagnostics:

      1) Trips vs temperature
      2) Trips vs wind chill
      3) Trips vs precipitation
      4) Trips vs hour of day
      5) Price metrics vs weather
      6) Driver pay vs fare vs weather
      + all hour-of-week charts (requests + base fare)
    """
    plot_trips_vs_temperature(df, save=save_plots, out_dir=out_dir)
    plot_trips_vs_wind_chill(df, save=save_plots, out_dir=out_dir)
    plot_trips_vs_precipitation(df, save=save_plots, out_dir=out_dir)

    # Hour-of-day pattern (uses broken axis when curfew=True)
    plot_requests_by_hour_of_day(df, save=save_plots, out_dir=out_dir, curfew=curfew)

    plot_price_metrics_vs_weather(df, save=save_plots, out_dir=out_dir)
    plot_driver_pay_vs_fare_vs_weather(df, save=save_plots, out_dir=out_dir)

    # Hour-of-week charts (extend here as we add more)
    plot_requests_by_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)
    plot_base_fare_by_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)
    plot_fare_rate_vs_trips_hour_of_week(df, save=save_plots, out_dir=out_dir, curfew=curfew)


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
    metric_col: str = "request_count",
    est_col: str | None = None,          # <-- NEW
    shade_hours: tuple[int, int] | None = None,
    curfew: bool = True,
    save_plots: bool = False,
) -> None:
    """
    Convenience wrapper:
      - Reads processed file
      - Applies optional curfew filter (drop hours 2–6)
      - Runs the day-vs-weekday comparison plot on a chosen metric.
    """
    input_path = PROCESSED_DIR / file_name
    df = _read_input(input_path)

    if curfew and "request_hour" in df.columns:
        df = df[~df["request_hour"].isin(CURFEW_HOURS)].copy()

    if DEFAULT_TIME_COL in df.columns:
        df = df.sort_values(DEFAULT_TIME_COL)

    plot_metric_day_vs_weekday_avg(
        df=df,
        target_date=target_date,
        metric_col=metric_col,
        compare_dow1=compare_dow1,
        compare_dow2=compare_dow2,
        est_col=est_col,                  # <-- pass through
        shade_hours=shade_hours,
        time_col=DEFAULT_TIME_COL,
        hour_col="request_hour",
        save=save_plots,
        out_dir=None,
    )
    print(
        f"Day vs weekday comparison complete for {target_date} "
        f"(metric={metric_col}) from: {input_path}"
    )



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA plotting for hourly TLC + weather data.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input dataframe (.parquet or .csv). "
             "Relative to project root or absolute.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="If set, save plots to <PROJECT_ROOT>/plots instead of showing them.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for output directory. "
             "If omitted and --save-plots is set, defaults to <PROJECT_ROOT>/plots.",
    )

    return parser.parse_args()


def main(file_name: str, save_plots: bool = False, curfew: bool = True, core: bool = False):
    """
    Run EDA given just a file name; the directory is PROCESSED_DIR.

    Examples (Spyder):
        main("fhvhv_lga_hourly_with_weather_2025.parquet", True, curfew=True, core=True)
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
    CURFEW = False    # True => drop hours 2–6
    CORE = False      # True => core diagnostics only; False => full EDA

    main(FILE_NAME, SAVE_PLOTS, CURFEW, CORE)