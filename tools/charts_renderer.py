import pandas as pd
import plotly.express as px

from tools.chart_schema import is_datetime_series
from tools.charts_model import ChartSpec

MAX_PIE_CATEGORIES = 12
MAX_BAR_CATEGORIES = 20
COUNT_COLUMN = "count"

# Pandas 3+ month/quarter/year-end aliases (legacy M/Q/Y still accepted from the LLM)
LEGACY_FREQ_MAP = {"M": "ME", "Q": "QE", "Y": "YE"}


def resolve_time_freq(freq: str | None) -> str:
    raw = freq or "ME"
    return LEGACY_FREQ_MAP.get(raw, raw)


def _parse_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _limit_categories(df: pd.DataFrame, category_col: str, value_col: str, limit: int) -> pd.DataFrame:
    ranked = df.groupby(category_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    top = ranked.head(limit).index.tolist()
    plot_df = df.copy()
    plot_df[category_col] = plot_df[category_col].where(plot_df[category_col].isin(top), "Other")
    return plot_df.groupby(category_col, dropna=False)[value_col].sum().reset_index()


def prepare_plot_data(df: pd.DataFrame, chart: ChartSpec) -> tuple[pd.DataFrame | None, str | None]:
    """Aggregate and shape data before plotting. Returns (dataframe, y_column)."""
    if chart.x not in df.columns:
        return None, None

    plot_df = df.copy()
    agg = chart.aggregation
    freq = resolve_time_freq(chart.time_freq)

    if agg == "count":
        x_series = plot_df[chart.x]
        if is_datetime_series(x_series):
            plot_df[chart.x] = _parse_datetime(plot_df[chart.x])
            grouped = (
                plot_df.groupby(pd.Grouper(key=chart.x, freq=freq))
                .size()
                .reset_index(name=COUNT_COLUMN)
            )
            return grouped.dropna(subset=[chart.x]), COUNT_COLUMN

        grouped = plot_df.groupby(chart.x, dropna=False).size().reset_index(name=COUNT_COLUMN)
        if chart.chart_type == "bar" and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, chart.x, COUNT_COLUMN, MAX_BAR_CATEGORIES)
        if chart.chart_type == "pie" and len(grouped) > MAX_PIE_CATEGORIES:
            grouped = _limit_categories(grouped, chart.x, COUNT_COLUMN, MAX_PIE_CATEGORIES)
        return grouped, COUNT_COLUMN

    if agg in ("sum", "mean") and chart.y and chart.y in plot_df.columns:
        x_series = plot_df[chart.x]
        if is_datetime_series(x_series):
            plot_df[chart.x] = _parse_datetime(plot_df[chart.x])
            grouped = plot_df.groupby(pd.Grouper(key=chart.x, freq=freq))[chart.y]
            if agg == "sum":
                result = grouped.sum().reset_index()
            else:
                result = grouped.mean().reset_index()
            return result.dropna(subset=[chart.x]), chart.y

        if agg == "sum":
            grouped = plot_df.groupby(chart.x, dropna=False)[chart.y].sum().reset_index()
        else:
            grouped = plot_df.groupby(chart.x, dropna=False)[chart.y].mean().reset_index()

        if chart.chart_type == "bar" and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, chart.x, chart.y, MAX_BAR_CATEGORIES)
        return grouped, chart.y

    if chart.y and chart.y in plot_df.columns:
        if is_datetime_series(plot_df[chart.x]):
            plot_df[chart.x] = _parse_datetime(plot_df[chart.x])
            plot_df = plot_df.sort_values(chart.x)
        if chart.chart_type == "pie" and plot_df[chart.x].nunique() > MAX_PIE_CATEGORIES:
            limited = (
                plot_df.groupby(chart.x, dropna=False)[chart.y]
                .sum()
                .reset_index()
            )
            return _limit_categories(limited, chart.x, chart.y, MAX_PIE_CATEGORIES), chart.y
        if chart.chart_type == "bar" and plot_df[chart.x].nunique() > MAX_BAR_CATEGORIES:
            limited = (
                plot_df.groupby(chart.x, dropna=False)[chart.y]
                .sum()
                .reset_index()
            )
            return _limit_categories(limited, chart.x, chart.y, MAX_BAR_CATEGORIES), chart.y
        return plot_df, chart.y

    return None, None


def render_chart(df, chart: ChartSpec):
    plot_df, y_col = prepare_plot_data(df, chart)
    if plot_df is None or y_col is None:
        return None

    if chart.chart_type == "line":
        fig = px.line(plot_df, x=chart.x, y=y_col, title=chart.title)

    elif chart.chart_type == "bar":
        fig = px.bar(plot_df, x=chart.x, y=y_col, title=chart.title)

    elif chart.chart_type == "scatter":
        fig = px.scatter(plot_df, x=chart.x, y=y_col, title=chart.title)

    elif chart.chart_type == "pie":
        fig = px.pie(plot_df, names=chart.x, values=y_col, title=chart.title)

    else:
        return None

    return fig
