"""Schema builders and column heuristics for chart recommendation and rendering."""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

# Column names that likely hold record IDs, not metrics
ID_COLUMN_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|emp.?id|employee.?id|user.?id|"
    r"order.?id|customer.?id|account.?id|record.?id)",
    re.IGNORECASE,
)

TREND_TITLE_PATTERN = re.compile(
    r"(headcount|hiring|trend|over time|growth|momentum|cumulative|volume)",
    re.IGNORECASE,
)

METRIC_TREND_TITLE_PATTERN = re.compile(
    r"(api|hits|usage|users|revenue|engagement|feed|spatial|incremental|volume)",
    re.IGNORECASE,
)

HEADCOUNT_TITLE_PATTERN = re.compile(
    r"(headcount|hiring|employee|staff|workforce|join)",
    re.IGNORECASE,
)

# Synthetic datetime column created when Year and Month exist separately
PERIOD_COLUMN = "_period"

# Pie charts are hard to read beyond a handful of slices
MAX_PIE_CATEGORIES = 6

# min/max ratio above this threshold suggests a log y-axis
LOG_SCALE_RATIO_THRESHOLD = 100


def to_json_safe(obj):
    """Convert numpy/pandas scalars so json.dumps works reliably."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {to_json_safe(key): to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(item) for item in obj]
    return str(obj)


def looks_like_id(column_name: str) -> bool:
    """Heuristic: column name matches common ID patterns."""
    normalized = column_name.strip().replace(" ", "")
    return bool(ID_COLUMN_PATTERN.search(normalized) or ID_COLUMN_PATTERN.search(column_name))


def is_numeric_series(series: pd.Series) -> bool:
    """True if the series has a numeric dtype."""
    return pd.api.types.is_numeric_dtype(series)


def find_year_month_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detect separate Year and Month columns by name."""
    year_col = month_col = None
    for col in df.columns:
        name = col.lower().strip()
        if name in ("year", "yr") or name.endswith(" year"):
            year_col = col
        if name in ("month", "mo", "mnth") or name.endswith(" month"):
            month_col = col
    return year_col, month_col


def find_year_column(df: pd.DataFrame) -> str | None:
    """Return the best year column for filtering, if any."""
    year_col, _ = find_year_month_columns(df)
    if year_col:
        return year_col
    for col in df.columns:
        if is_year_like_column(col, df[col]):
            return col
    return None


def get_available_years(df: pd.DataFrame, time_col: str) -> list[int]:
    """Sorted distinct years from a year or datetime column."""
    if time_col not in df.columns:
        return []
    series = df[time_col]
    if is_datetime_series(series):
        years = pd.to_datetime(series, errors="coerce").dt.year.dropna().astype(int)
        return sorted(years.unique().tolist())
    years = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


TIME_GRANULARITY_FREQ = {
    "5min": "5min",
    "min": "min",
    "hour": "h",
    "day": "D",
    "week": "W-MON",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}

TIME_GRANULARITY_LABELS = {
    "5min": "5-minute period",
    "min": "Minute",
    "hour": "Hour",
    "day": "Day",
    "week": "Week",
    "month": "Month",
    "quarter": "Quarter",
    "year": "Year",
}

TIME_GRANULARITY_END_OFFSET = {
    "5min": pd.Timedelta(minutes=5),
    "min": pd.Timedelta(minutes=1),
    "hour": pd.Timedelta(hours=1),
    "day": pd.Timedelta(days=1),
    "week": pd.Timedelta(weeks=1),
    "month": pd.DateOffset(months=1),
    "quarter": pd.DateOffset(months=3),
    "year": pd.DateOffset(years=1),
}


@dataclass(frozen=True)
class TimeSliderConfig:
    """Dynamic time slider: granularity and selectable periods from the dataset."""

    granularity: str
    periods: tuple[pd.Timestamp, ...]
    labels: tuple[str, ...]

    @property
    def slider_label(self) -> str:
        return TIME_GRANULARITY_LABELS.get(self.granularity, "Time period")


def resolve_timeline_series(df: pd.DataFrame, chart) -> pd.Series | None:
    """Build a datetime series aligned to df rows for time filtering."""
    year_col, month_col = find_year_month_columns(df)
    if year_col and month_col:
        return build_period_series(df, year_col, month_col)

    time_col = resolve_time_column(df, chart)
    if time_col and time_col in df.columns:
        series = df[time_col]
        if is_datetime_series(series):
            return pd.to_datetime(series, errors="coerce")
        if is_year_like_column(time_col, series):
            years = pd.to_numeric(series, errors="coerce")
            return pd.to_datetime(years.astype("Int64").astype(str) + "-01-01", errors="coerce")

    return None


def _granularity_candidates(span_seconds: float) -> list[str]:
    """Rank granularities to try for a given time span."""
    if span_seconds <= 6 * 3600:
        return ["5min", "min", "hour", "day"]
    if span_seconds <= 2 * 86400:
        return ["hour", "day", "week"]
    if span_seconds <= 21 * 86400:
        return ["day", "week", "month"]
    if span_seconds <= 180 * 86400:
        return ["week", "month", "day"]
    if span_seconds <= 2 * 365 * 86400:
        return ["month", "week", "quarter"]
    if span_seconds <= 8 * 365 * 86400:
        return ["month", "quarter", "year"]
    return ["quarter", "year", "month"]


def bucket_timeline(timeline: pd.Series, granularity: str) -> pd.Series:
    """Floor timestamps to the start of each granularity bucket."""
    parsed = pd.to_datetime(timeline, errors="coerce")
    if granularity == "month":
        return parsed.dt.to_period("M").dt.to_timestamp()
    if granularity == "quarter":
        return parsed.dt.to_period("Q").dt.to_timestamp()
    if granularity == "year":
        return parsed.dt.to_period("Y").dt.to_timestamp()
    return parsed.dt.floor(TIME_GRANULARITY_FREQ[granularity])


def periods_from_timeline(timeline: pd.Series, granularity: str) -> list[pd.Timestamp]:
    """Distinct time buckets present in the data for a granularity."""
    buckets = bucket_timeline(timeline.dropna(), granularity).dropna().unique()
    return sorted(pd.Timestamp(value) for value in buckets)


def pick_time_granularity(timeline: pd.Series) -> tuple[str, list[pd.Timestamp]]:
    """Choose slider step size based on how the timeline is spread."""
    valid = pd.to_datetime(timeline, errors="coerce").dropna()
    if valid.empty:
        return "month", []

    min_ts, max_ts = valid.min(), valid.max()
    span_seconds = (max_ts - min_ts).total_seconds()
    if span_seconds <= 0:
        return "day", periods_from_timeline(valid, "day")

    candidates = _granularity_candidates(span_seconds)
    for granularity in candidates:
        periods = periods_from_timeline(valid, granularity)
        if 2 <= len(periods) <= 72:
            return granularity, periods

    for granularity in reversed(candidates):
        periods = periods_from_timeline(valid, granularity)
        if len(periods) >= 2:
            return granularity, periods

    return candidates[0], periods_from_timeline(valid, candidates[0])


def format_time_period_label(timestamp: pd.Timestamp, granularity: str) -> str:
    """Human-readable label for a slider step."""
    ts = pd.Timestamp(timestamp)
    if granularity in ("min", "5min"):
        return ts.strftime("%b %d, %Y %H:%M")
    if granularity == "hour":
        return ts.strftime("%b %d, %Y %H:%M")
    if granularity == "day":
        return ts.strftime("%b %d, %Y")
    if granularity == "week":
        return ts.strftime("Week of %b %d, %Y")
    if granularity == "month":
        return ts.strftime("%b %Y")
    if granularity == "quarter":
        return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
    if granularity == "year":
        return str(ts.year)
    return ts.strftime("%Y-%m-%d")


def period_end_exclusive(start: pd.Timestamp, granularity: str) -> pd.Timestamp:
    """Exclusive end of the bucket that begins at start."""
    return pd.Timestamp(start) + TIME_GRANULARITY_END_OFFSET[granularity]


def filter_df_to_time_period(
    df: pd.DataFrame,
    timeline: pd.Series,
    period_start: pd.Timestamp,
    granularity: str,
) -> pd.DataFrame:
    """Keep rows whose timeline value falls in the selected period bucket."""
    start = pd.Timestamp(period_start)
    end = period_end_exclusive(start, granularity)
    parsed = pd.to_datetime(timeline, errors="coerce")
    mask = (parsed >= start) & (parsed < end)
    return df.loc[mask.fillna(False)].copy()


def build_time_slider_config(df: pd.DataFrame, chart) -> TimeSliderConfig | None:
    """Build a dynamic time slider config when category charts span multiple periods."""
    timeline = resolve_timeline_series(df, chart)
    if timeline is None:
        return None

    valid = timeline.dropna()
    if len(valid) < 2:
        return None

    time_col = resolve_time_column(df, chart)
    if time_col and chart.group_by == time_col:
        return None

    granularity, periods = pick_time_granularity(valid)
    if len(periods) < 2:
        return None

    labels = tuple(format_time_period_label(period, granularity) for period in periods)
    return TimeSliderConfig(
        granularity=granularity,
        periods=tuple(periods),
        labels=labels,
    )


def resolve_time_column(df: pd.DataFrame, chart) -> str | None:
    """Return the column used for year filtering, if any."""
    if chart.time_column and chart.time_column in df.columns:
        if is_time_filter_column(chart.time_column, df[chart.time_column]):
            return chart.time_column
    year_col = find_year_column(df)
    if year_col:
        return year_col
    return None


def supports_time_slider(df: pd.DataFrame, chart) -> TimeSliderConfig | None:
    """Return slider config when a dynamic time filter applies to this chart."""
    if chart.chart_type not in ("bar", "line"):
        return None

    # Time on the x-axis already — a period slider would collapse the trend
    if is_time_series_chart(df, chart):
        return None

    return build_time_slider_config(df, chart)


def supports_legend_filter(df: pd.DataFrame, chart) -> bool:
    """True when the chart legend can toggle multiple series."""
    if chart.chart_type not in ("line", "bar", "scatter"):
        return False
    if not chart.group_by or chart.group_by not in df.columns:
        return False
    return df[chart.group_by].nunique(dropna=True) > 1


def is_year_like_column(column: str, series: pd.Series) -> bool:
    """True if the column name or value range looks like calendar years."""
    if "year" in column.lower():
        return True
    if is_numeric_series(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return non_null.min() >= 1900 and non_null.max() <= 2100 and non_null.nunique() <= 40
    return False


def is_month_like_column(column: str) -> bool:
    """True if the column name looks like a month field."""
    name = column.lower().strip()
    return name in ("month", "mo", "mnth") or name.endswith(" month")


def build_period_series(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    """Combine Year + Month into a datetime series (first of month)."""
    years = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    months = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    labels = years.astype(str) + "-" + months.astype(str).str.zfill(2) + "-01"
    return pd.to_datetime(labels, errors="coerce")


def title_implies_metric_trend(title: str) -> bool:
    """Chart title suggests summing a metric over time (not row counts)."""
    return bool(METRIC_TREND_TITLE_PATTERN.search(title))


def title_implies_headcount_trend(title: str) -> bool:
    """Chart title suggests counting rows (hiring/headcount) over time."""
    return bool(HEADCOUNT_TITLE_PATTERN.search(title))


def supports_trend_aggregation_toggle(chart) -> bool:
    """True for metric line trends where the user can switch sum vs mean."""
    return (
        chart.chart_type == "line"
        and chart.aggregation in ("sum", "mean")
        and chart.y is not None
    )


def is_time_series_chart(df: pd.DataFrame, chart) -> bool:
    """True when the chart x-axis represents time (year, month, or date)."""
    if chart.x not in df.columns:
        return False
    year_col, month_col = find_year_month_columns(df)
    if chart.x in (year_col, month_col):
        return True
    if is_datetime_series(df[chart.x]):
        return True
    return is_year_like_column(chart.x, df[chart.x])


def supports_trend_chart_controls(df: pd.DataFrame, chart) -> bool:
    """True for time-series line/bar charts that expose interactive trend controls."""
    return chart.chart_type in ("line", "bar") and is_time_series_chart(df, chart)


TREND_CHART_TYPES = ["line", "bar"]


def trend_metric_options(df: pd.DataFrame, chart) -> list[str]:
    """Numeric columns available as the trend y-axis metric."""
    year_col, month_col = find_year_month_columns(df)
    skip = {c for c in (year_col, month_col, chart.x) if c}
    return [
        col
        for col in df.columns
        if col not in skip
        and is_numeric_series(df[col])
        and not looks_like_id(col)
        and not is_year_like_column(col, df[col])
        and not is_month_like_column(col)
    ]


def trend_group_options(df: pd.DataFrame, chart, y_column: str | None = None) -> list[str]:
    """Categorical columns for comparing groups; first option is all groups combined."""
    year_col, month_col = find_year_month_columns(df)
    skip = {c for c in (year_col, month_col, chart.x, y_column or chart.y) if c}
    options = ["All groups"]
    for col in df.columns:
        if col in skip or looks_like_id(col):
            continue
        if is_numeric_series(df[col]) or is_datetime_series(df[col]):
            continue
        if is_year_like_column(col, df[col]) or is_month_like_column(col):
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 20:
            options.append(col)
    return options


_TITLE_BY_GROUP_PATTERN = re.compile(r"\s+by\s+.+$", re.IGNORECASE)

_GENERIC_METRIC_TITLE_PATTERN = re.compile(
    r"\b(revenue|sales|engagement|usage|users|user count|api hits|volume|performance|metrics?)\b",
    re.IGNORECASE,
)

_TIME_FREQ_GRANULARITY = {
    "D": "Daily",
    "W": "Weekly",
    "ME": "Monthly",
    "M": "Monthly",
    "QE": "Quarterly",
    "Q": "Quarterly",
    "YE": "Yearly",
    "Y": "Yearly",
}


def humanize_column_name(column: str) -> str:
    """Turn a dataset column name into a readable title fragment."""
    text = column.strip().replace("_", " ")
    if text != text.lower() and text != text.upper():
        return text
    return text.title()


def _title_mentions_metric(title: str, metric_label: str) -> bool:
    """True when the title already names the y-axis metric."""
    title_lower = title.lower()
    metric_lower = metric_label.lower()
    if metric_lower in title_lower:
        return True
    for token in re.split(r"[\s_]+", metric_lower):
        if len(token) > 2 and token in title_lower:
            return True
    return False


def _time_granularity_label(chart, df: pd.DataFrame) -> str | None:
    """Best-effort time bucket label (Monthly, Quarterly, etc.) for trend titles."""
    if chart.time_freq and chart.time_freq in _TIME_FREQ_GRANULARITY:
        return _TIME_FREQ_GRANULARITY[chart.time_freq]
    year_col, month_col = find_year_month_columns(df)
    if month_col and chart.x in (year_col, month_col):
        return "Monthly"
    if chart.x in df.columns and is_datetime_series(df[chart.x]):
        return None
    return None


def sync_chart_title_from_y(chart, df: pd.DataFrame) -> str:
    """Rewrite LLM titles so the metric name matches chart.y."""
    if not chart.y:
        return chart.title.strip()

    y_label = humanize_column_name(chart.y)
    base = _TITLE_BY_GROUP_PATTERN.sub("", chart.title).strip()
    if _title_mentions_metric(base, y_label):
        return base

    if is_time_series_chart(df, chart) and chart.chart_type in ("line", "bar"):
        granularity = _time_granularity_label(chart, df)
        if granularity:
            return f"{granularity} {y_label} Trend"
        return f"{y_label} Trend"

    replaced, count = _GENERIC_METRIC_TITLE_PATTERN.subn(y_label, base, count=1)
    if count:
        return replaced

    if chart.chart_type in ("bar", "pie", "scatter") and chart.x in df.columns:
        x_label = humanize_column_name(chart.x)
        if chart.chart_type == "scatter":
            return f"{y_label} vs {x_label}"
        return f"{y_label} by {x_label}"

    return base


def chart_display_title(chart) -> str:
    """Title reflecting the active Compare groups selection."""
    base = chart.title.strip()
    if chart.group_by:
        if _TITLE_BY_GROUP_PATTERN.search(base):
            return _TITLE_BY_GROUP_PATTERN.sub(f" by {chart.group_by}", base)
        return f"{base} by {chart.group_by}"
    return _TITLE_BY_GROUP_PATTERN.sub("", base).strip()


def chart_display_reason(chart) -> str:
    """Caption reflecting the active Compare groups selection."""
    if not chart.group_by:
        return chart.reason
    reason = chart.reason
    reason = re.sub(r"\bacross themes\b", f"across {chart.group_by}", reason, flags=re.IGNORECASE)
    reason = re.sub(r"\beach theme's\b", f"each {chart.group_by}'s", reason, flags=re.IGNORECASE)
    reason = re.sub(r"\bthemes\b", chart.group_by, reason, flags=re.IGNORECASE)
    if _TITLE_BY_GROUP_PATTERN.search(reason):
        return _TITLE_BY_GROUP_PATTERN.sub(f" by {chart.group_by}", reason)
    return reason


def looks_like_rating_column(column: str, series: pd.Series) -> bool:
    """True for bounded scores/ratings where mean is more meaningful than sum."""
    if not is_numeric_series(series):
        return False
    name = column.lower()
    if any(k in name for k in ("csat", "rating", "score", "satisfaction", "nps")):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    return non_null.max() <= 10 and non_null.min() >= 0 and non_null.nunique() <= 10


def values_suggest_log_scale(series: pd.Series) -> bool:
    """True when positive values span a large range (orders of magnitude)."""
    if not is_numeric_series(series):
        return False
    positive = series.dropna()
    positive = positive[positive > 0]
    if len(positive) < 2:
        return False
    vmin, vmax = float(positive.min()), float(positive.max())
    return (vmax / vmin) >= LOG_SCALE_RATIO_THRESHOLD


def find_metric_column_for_title(df: pd.DataFrame, title: str) -> str | None:
    """Guess the best numeric metric column from chart title and df columns."""
    title_lower = title.lower()
    year_col, month_col = find_year_month_columns(df)
    skip = {c for c in (year_col, month_col) if c}

    # Prefer columns whose name appears in the chart title
    direct = []
    for col in df.columns:
        if col in skip or not is_numeric_series(df[col]) or looks_like_id(col):
            continue
        col_lower = col.lower()
        if col_lower in title_lower:
            direct.append(col)
            continue
        for token in re.split(r"[\s_]+", col_lower):
            if len(token) > 3 and token in title_lower:
                direct.append(col)
                break

    if direct:
        return direct[0]

    # Fallback: known metric keywords in column names
    for col in df.columns:
        if col in skip or not is_numeric_series(df[col]):
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in ("api hits", "hits", "users", "revenue", "feed")):
            return col

    numeric_cols = [
        c
        for c in df.columns
        if c not in skip and is_numeric_series(df[c]) and not looks_like_id(c)
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    return None


def is_datetime_series(series: pd.Series) -> bool:
    """True for datetime dtypes or object columns that mostly parse as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        parsed = pd.to_datetime(series.head(20), errors="coerce")
        return parsed.notna().mean() >= 0.8
    return False


def is_time_filter_column(column: str, series: pd.Series) -> bool:
    """True when a column can drive the time slider (Year field or datetime)."""
    if is_year_like_column(column, series):
        return True
    return is_datetime_series(series)


def column_metadata(df: pd.DataFrame, column: str) -> dict:
    """Per-column stats sent to the LLM in the chart recommendation schema."""
    series = df[column]
    nunique = int(series.nunique(dropna=True))
    metadata = {
        "name": column,
        "dtype": str(series.dtype),
        "nunique": nunique,
        "is_numeric": bool(is_numeric_series(series)),
        "is_datetime": bool(is_datetime_series(series)),
        "looks_like_id": bool(looks_like_id(column)),
        "missing": int(series.isna().sum()),
    }
    if metadata["is_datetime"]:
        parsed = pd.to_datetime(series, errors="coerce")
        valid = parsed.dropna()
        if not valid.empty:
            metadata["min"] = valid.min().isoformat()
            metadata["max"] = valid.max().isoformat()
    elif metadata["is_numeric"]:
        metadata["min"] = float(series.min()) if series.notna().any() else None
        metadata["max"] = float(series.max()) if series.notna().any() else None
        metadata["is_year_like"] = bool(is_year_like_column(column, series))
        metadata["is_month_like"] = bool(is_month_like_column(column))
        metadata["suggests_log_scale"] = bool(values_suggest_log_scale(series))
    else:
        metadata["is_year_like"] = False
        metadata["is_month_like"] = False
        if nunique <= 15:
            metadata["top_values"] = {
                str(key): int(value)
                for key, value in series.value_counts(dropna=True).head(5).items()
            }
    return metadata


def build_chart_schema(df: pd.DataFrame) -> dict:
    """Full dataset description for the chart recommender prompt (columns + hints + samples)."""
    columns = [column_metadata(df, col) for col in df.columns]
    year_col, month_col = find_year_month_columns(df)
    hints: dict = {}
    if year_col and month_col:
        hints["year_month_pair"] = [year_col, month_col]
        hints["monthly_trend_instruction"] = (
            f"For monthly trends use x={year_col!r}, y=<metric column>, "
            f"aggregation=sum (NOT count on Year). Renderer combines {year_col} + {month_col}."
        )
    metric_cols = [
        c["name"]
        for c in columns
        if c.get("is_numeric") and not c.get("looks_like_id") and not c.get("is_year_like")
    ]
    if metric_cols:
        hints["metric_columns"] = metric_cols

    hints["pie_chart_max_categories"] = MAX_PIE_CATEGORIES
    hints["pie_chart_rule"] = (
        f"Do NOT recommend pie charts when x has more than {MAX_PIE_CATEGORIES} groups "
        f"(check nunique in columns); use a bar chart instead."
    )
    hints["log_scale_rule"] = (
        "Set log_y=true on line, bar, or scatter charts when y values span a large range "
        f"(roughly {LOG_SCALE_RATIO_THRESHOLD}x or more, or suggests_log_scale is true on the metric)."
    )

    year_col = find_year_column(df)
    if year_col:
        years = get_available_years(df, year_col)
        hints["year_column"] = year_col
        hints["available_years"] = years
        hints["time_slider_rule"] = (
            f"When comparing categories or groups over time, set time_column={year_col!r} "
            "or a datetime column. The dashboard picks slider steps (day/week/month/etc.) "
            "based on how the timeline is spread."
        )

    group_candidates = [
        c["name"]
        for c in columns
        if not c.get("is_numeric")
        and not c.get("is_datetime")
        and not c.get("looks_like_id")
        and not c.get("is_year_like")
        and 2 <= c.get("nunique", 0) <= 12
    ]
    if group_candidates:
        hints["group_by_candidates"] = group_candidates
        hints["comparison_chart_rule"] = (
            "For group comparisons use bar or line with group_by (series column), "
            'bar_mode "group" for clustered/side-by-side bars or "stack" for stacked bars/columns.'
        )

    schema = {
        "rows": int(df.shape[0]),
        "columns": columns,
        "hints": hints,
        "sample_rows": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
    return to_json_safe(schema)
