"""Fix and validate LLM chart specs before rendering."""

import pandas as pd

from tools.chart_schema import (
    MAX_PIE_CATEGORIES,
    find_metric_column_for_title,
    find_year_column,
    find_year_month_columns,
    get_available_years,
    is_datetime_series,
    is_numeric_series,
    is_time_filter_column,
    is_time_series_chart,
    is_year_like_column,
    looks_like_id,
    sync_chart_title_from_y,
    title_implies_headcount_trend,
    title_implies_metric_trend,
    values_suggest_log_scale,
)
from tools.charts_model import ChartSpec

SUPPORTED_CHART_TYPES = {"line", "bar", "pie", "scatter"}
MAX_BAR_CATEGORIES = 20
MAX_CATEGORICAL_Y_FOR_LINE = 30


def normalize_chart_spec(df: pd.DataFrame, chart: ChartSpec) -> ChartSpec:
    """Fix common LLM mistakes before validation and rendering."""
    updates: dict = {}

    y_is_id = chart.y is not None and looks_like_id(chart.y)
    x_is_date = chart.x in df.columns and is_datetime_series(df[chart.x])
    x_is_year = chart.x in df.columns and is_year_like_column(chart.x, df[chart.x])
    year_col, month_col = find_year_month_columns(df)
    metric_col = find_metric_column_for_title(df, chart.title)

    # Metric trends (API hits, revenue, etc.) should sum a metric, not count rows
    if title_implies_metric_trend(chart.title):
        if chart.aggregation == "count" or (x_is_year and chart.aggregation == "none"):
            if metric_col:
                updates["aggregation"] = "sum"
                updates["y"] = metric_col
                if year_col and month_col and chart.x in (year_col, month_col):
                    updates["x"] = year_col
        elif chart.aggregation == "none" and metric_col and chart.chart_type == "line":
            updates["aggregation"] = "sum"
            updates["y"] = metric_col

    # Headcount/hiring charts count rows over time instead of summing an ID column
    if title_implies_headcount_trend(chart.title) and not title_implies_metric_trend(chart.title):
        if x_is_date and chart.aggregation == "none":
            updates["aggregation"] = "count"
            updates["y"] = None
            if not chart.time_freq:
                updates["time_freq"] = "ME"

    if y_is_id or (title_implies_headcount_trend(chart.title) and x_is_date and chart.aggregation == "none"):
        if not title_implies_metric_trend(chart.title):
            updates["aggregation"] = "count"
            updates["y"] = None
            if x_is_date and not chart.time_freq:
                updates["time_freq"] = "ME"

    if chart.aggregation == "count" and chart.y and looks_like_id(chart.y):
        updates["y"] = None

    if chart.aggregation in ("sum", "mean") and chart.y and looks_like_id(chart.y):
        if metric_col:
            updates["y"] = metric_col
        else:
            updates["aggregation"] = "count"
            updates["y"] = None

    if (
        chart.aggregation == "count"
        and x_is_year
        and metric_col
        and title_implies_metric_trend(chart.title)
    ):
        updates["aggregation"] = "sum"
        updates["y"] = metric_col

    if chart.chart_type == "pie" and chart.x in df.columns:
        if df[chart.x].nunique(dropna=True) > MAX_PIE_CATEGORIES:
            updates["chart_type"] = "bar"

    if (
        chart.chart_type in ("line", "bar", "scatter")
        and chart.y
        and chart.y in df.columns
        and not chart.log_y
        and values_suggest_log_scale(df[chart.y])
    ):
        updates["log_y"] = True

    if chart.chart_type == "bar" and chart.group_by and not chart.bar_mode:
        updates["bar_mode"] = "group"

    if chart.group_by and chart.group_by not in df.columns:
        updates["group_by"] = None
        updates["bar_mode"] = None
    elif chart.group_by:
        if chart.group_by == chart.x or chart.group_by == chart.y:
            updates["group_by"] = None
            updates["bar_mode"] = None
        elif looks_like_id(chart.group_by):
            updates["group_by"] = None
            updates["bar_mode"] = None

    year_col = find_year_column(df)
    if (
        year_col
        and not chart.time_column
        and not is_time_series_chart(df, chart)
        and chart.chart_type in ("bar", "line")
        and chart.x != year_col
        and len(get_available_years(df, year_col)) >= 2
    ):
        updates["time_column"] = year_col

    if chart.time_column and is_time_series_chart(df, chart):
        updates["time_column"] = None

    if chart.time_column:
        if chart.time_column not in df.columns:
            updates["time_column"] = find_year_column(df)
        elif not is_time_filter_column(chart.time_column, df[chart.time_column]):
            updates["time_column"] = find_year_column(df)

    if updates:
        chart = chart.model_copy(update=updates)

    if chart.bar_mode and (not chart.group_by or chart.chart_type != "bar"):
        chart = chart.model_copy(update={"bar_mode": None})

    synced_title = sync_chart_title_from_y(chart, df)
    if synced_title != chart.title:
        chart = chart.model_copy(update={"title": synced_title})

    return chart


def validate_chart(df: pd.DataFrame, chart: ChartSpec) -> tuple[bool, str | None]:
    """Return (True, None) if the chart can be rendered, else (False, error message)."""
    if chart.chart_type not in SUPPORTED_CHART_TYPES:
        return False, f"Unsupported chart type: {chart.chart_type}"

    if chart.x not in df.columns:
        return False, f"Column '{chart.x}' is not in the dataset."

    if chart.group_by:
        if chart.group_by not in df.columns:
            return False, f"Column '{chart.group_by}' is not in the dataset."
        if chart.group_by == chart.x:
            return False, "group_by must differ from x."
        if chart.y and chart.group_by == chart.y:
            return False, "group_by must differ from y."
        if looks_like_id(chart.group_by):
            return False, f"Column '{chart.group_by}' looks like an identifier, not a group."
        if chart.chart_type == "pie":
            return False, "Pie charts do not support group_by."

    if chart.bar_mode and chart.chart_type != "bar":
        return False, "bar_mode applies only to bar charts."

    if chart.time_column:
        if chart.time_column not in df.columns:
            return False, f"Column '{chart.time_column}' is not in the dataset."
        if not is_time_filter_column(chart.time_column, df[chart.time_column]):
            return False, (
                f"Column '{chart.time_column}' cannot be used for year filtering "
                "(use a Year column or a date/datetime column)."
            )

    if chart.aggregation in ("sum", "mean"):
        if not chart.y:
            return False, f"'{chart.aggregation}' aggregation requires a numeric y column."
        if chart.y not in df.columns:
            return False, f"Column '{chart.y}' is not in the dataset."
        if looks_like_id(chart.y):
            return False, f"Column '{chart.y}' looks like an identifier, not a metric."
        if not is_numeric_series(df[chart.y]):
            return False, f"Column '{chart.y}' must be numeric for {chart.aggregation}."

    if chart.aggregation == "none" and chart.chart_type in ("line", "scatter"):
        if not chart.y:
            return False, "Line and scatter charts need a y column or aggregation."
        if chart.y not in df.columns:
            return False, f"Column '{chart.y}' is not in the dataset."
        if looks_like_id(chart.y):
            return False, f"Column '{chart.y}' looks like an identifier, not a metric."
        if not is_numeric_series(df[chart.y]):
            nunique = df[chart.y].nunique(dropna=True)
            if nunique > MAX_CATEGORICAL_Y_FOR_LINE:
                return (
                    False,
                    f"Column '{chart.y}' has too many categories ({nunique}) for a line chart.",
                )

    if chart.aggregation == "none" and chart.chart_type == "bar":
        if not chart.y:
            return False, "Bar charts need a y column or aggregation."
        if chart.y not in df.columns:
            return False, f"Column '{chart.y}' is not in the dataset."
        if looks_like_id(chart.y):
            return False, f"Column '{chart.y}' looks like an identifier, not a metric."

    if chart.aggregation == "none" and chart.chart_type == "pie":
        if not chart.y:
            return False, "Pie charts need a values column (y) or use aggregation 'count'."
        if chart.y not in df.columns:
            return False, f"Column '{chart.y}' is not in the dataset."
        if looks_like_id(chart.y):
            return False, f"Column '{chart.y}' looks like an identifier, not a metric."

    if chart.aggregation == "count" and chart.x not in df.columns:
        return False, f"Column '{chart.x}' is not in the dataset."

    if (
        chart.aggregation == "count"
        and chart.x in df.columns
        and is_year_like_column(chart.x, df[chart.x])
        and title_implies_metric_trend(chart.title)
    ):
        return (
            False,
            f"Use aggregation 'sum' with a metric column (e.g. API Hits), not row count by {chart.x}.",
        )

    # Pie charts need at most MAX_PIE_CATEGORIES groups (prefer bar beyond that)
    if chart.chart_type == "pie" and chart.x in df.columns:
        nunique = df[chart.x].nunique(dropna=True)
        if nunique > MAX_PIE_CATEGORIES:
            return (
                False,
                f"Column '{chart.x}' has {nunique} categories; pie charts need at most "
                f"{MAX_PIE_CATEGORIES} — use a bar chart instead.",
            )

    if chart.chart_type == "bar" and chart.aggregation in ("none", "count"):
        nunique = df[chart.x].nunique(dropna=True)
        if chart.aggregation == "none" and nunique > MAX_BAR_CATEGORIES:
            return (
                False,
                f"Column '{chart.x}' has {nunique} categories; bar charts need at most "
                f"{MAX_BAR_CATEGORIES} distinct x values.",
            )

    return True, None
