import pandas as pd

from tools.chart_schema import (
    TREND_TITLE_PATTERN,
    is_datetime_series,
    is_numeric_series,
    looks_like_id,
)
from tools.charts_model import ChartSpec

SUPPORTED_CHART_TYPES = {"line", "bar", "pie", "scatter"}
MAX_PIE_CATEGORIES = 12
MAX_BAR_CATEGORIES = 20
MAX_CATEGORICAL_Y_FOR_LINE = 30


def normalize_chart_spec(df: pd.DataFrame, chart: ChartSpec) -> ChartSpec:
    """Fix common LLM mistakes before validation and rendering."""
    updates: dict = {}

    y_is_id = chart.y is not None and looks_like_id(chart.y)
    x_is_date = chart.x in df.columns and is_datetime_series(df[chart.x])
    title_implies_trend = bool(TREND_TITLE_PATTERN.search(chart.title))

    if y_is_id or (title_implies_trend and x_is_date and chart.aggregation == "none"):
        updates["aggregation"] = "count"
        updates["y"] = None
        if x_is_date and not chart.time_freq:
            updates["time_freq"] = "ME"

    if chart.aggregation == "count" and chart.y and looks_like_id(chart.y):
        updates["y"] = None

    if chart.aggregation in ("sum", "mean") and chart.y and looks_like_id(chart.y):
        updates["aggregation"] = "count"
        updates["y"] = None
        if x_is_date:
            updates["time_freq"] = chart.time_freq or "ME"

    if updates:
        return chart.model_copy(update=updates)
    return chart


def validate_chart(df: pd.DataFrame, chart: ChartSpec) -> tuple[bool, str | None]:
    if chart.chart_type not in SUPPORTED_CHART_TYPES:
        return False, f"Unsupported chart type: {chart.chart_type}"

    if chart.x not in df.columns:
        return False, f"Column '{chart.x}' is not in the dataset."

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

    if chart.chart_type == "pie" and chart.aggregation == "none":
        nunique = df[chart.x].nunique(dropna=True)
        if nunique > MAX_PIE_CATEGORIES:
            return (
                False,
                f"Column '{chart.x}' has {nunique} categories; pie charts need at most "
                f"{MAX_PIE_CATEGORIES} (use aggregation or a grouped column).",
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
