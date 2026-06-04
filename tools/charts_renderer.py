import pandas as pd
import plotly.express as px

from tools.chart_schema import (
    PERIOD_COLUMN,
    build_period_series,
    find_year_month_columns,
    is_datetime_series,
    is_year_like_column,
)
from tools.charts_model import ChartSpec

MAX_PIE_CATEGORIES = 12
MAX_BAR_CATEGORIES = 20
COUNT_COLUMN = "count"

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


def _should_use_year_month_period(df: pd.DataFrame, chart: ChartSpec) -> bool:
    year_col, month_col = find_year_month_columns(df)
    if not year_col or not month_col:
        return False
    if chart.x not in (year_col, month_col):
        return False
    if chart.chart_type not in ("line", "bar"):
        return False
    return True


def _resolve_x_column(df: pd.DataFrame, chart: ChartSpec) -> tuple[pd.DataFrame, str]:
    plot_df = df.copy()
    if _should_use_year_month_period(plot_df, chart):
        year_col, month_col = find_year_month_columns(plot_df)
        plot_df[PERIOD_COLUMN] = build_period_series(plot_df, year_col, month_col)
        plot_df = plot_df.dropna(subset=[PERIOD_COLUMN])
        return plot_df, PERIOD_COLUMN

    if chart.x in plot_df.columns and is_year_like_column(chart.x, plot_df[chart.x]):
        plot_df[chart.x] = pd.to_numeric(plot_df[chart.x], errors="coerce").astype("Int64").astype(str)

    return plot_df, chart.x


def _sort_plot_df(plot_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    if x_col not in plot_df.columns:
        return plot_df
    if is_datetime_series(plot_df[x_col]) or pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        return plot_df.sort_values(x_col)
    return plot_df.sort_values(x_col, key=lambda s: pd.to_numeric(s, errors="coerce"))


def _apply_line_layout(fig, x_col: str, plot_df: pd.DataFrame) -> None:
    if x_col == PERIOD_COLUMN or is_datetime_series(plot_df[x_col]):
        fig.update_xaxes(tickformat="%b %Y", title="Month")
    elif x_col in plot_df.columns and is_year_like_column(x_col, plot_df[x_col]):
        fig.update_xaxes(type="category", title=x_col)


def prepare_plot_data(df: pd.DataFrame, chart: ChartSpec) -> tuple[pd.DataFrame | None, str | None]:
    if chart.x not in df.columns:
        return None, None

    plot_df, x_col = _resolve_x_column(df, chart)
    agg = chart.aggregation
    freq = resolve_time_freq(chart.time_freq)

    if agg == "count":
        x_series = plot_df[x_col]
        if is_datetime_series(x_series) or x_col == PERIOD_COLUMN:
            if x_col != PERIOD_COLUMN:
                plot_df[x_col] = _parse_datetime(plot_df[x_col])
            grouped = (
                plot_df.groupby(pd.Grouper(key=x_col, freq=freq))
                .size()
                .reset_index(name=COUNT_COLUMN)
            )
            return _sort_plot_df(grouped.dropna(subset=[x_col]), x_col), COUNT_COLUMN

        grouped = plot_df.groupby(x_col, dropna=False).size().reset_index(name=COUNT_COLUMN)
        if chart.chart_type == "bar" and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, COUNT_COLUMN, MAX_BAR_CATEGORIES)
        if chart.chart_type == "pie" and len(grouped) > MAX_PIE_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, COUNT_COLUMN, MAX_PIE_CATEGORIES)
        return _sort_plot_df(grouped, x_col), COUNT_COLUMN

    if agg in ("sum", "mean") and chart.y and chart.y in plot_df.columns:
        x_series = plot_df[x_col]
        if is_datetime_series(x_series) or x_col == PERIOD_COLUMN:
            if x_col != PERIOD_COLUMN:
                plot_df[x_col] = _parse_datetime(plot_df[x_col])
            grouped = plot_df.groupby(pd.Grouper(key=x_col, freq=freq))[chart.y]
            if agg == "sum":
                result = grouped.sum().reset_index()
            else:
                result = grouped.mean().reset_index()
            return _sort_plot_df(result.dropna(subset=[x_col]), x_col), chart.y

        if agg == "sum":
            grouped = plot_df.groupby(x_col, dropna=False)[chart.y].sum().reset_index()
        else:
            grouped = plot_df.groupby(x_col, dropna=False)[chart.y].mean().reset_index()

        if chart.chart_type == "bar" and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, chart.y, MAX_BAR_CATEGORIES)
        return _sort_plot_df(grouped, x_col), chart.y

    if chart.y and chart.y in plot_df.columns:
        if is_datetime_series(plot_df[x_col]):
            plot_df[x_col] = _parse_datetime(plot_df[x_col])
        plot_df = _sort_plot_df(plot_df, x_col)
        if chart.chart_type == "pie" and plot_df[x_col].nunique() > MAX_PIE_CATEGORIES:
            limited = plot_df.groupby(x_col, dropna=False)[chart.y].sum().reset_index()
            return _limit_categories(limited, x_col, chart.y, MAX_PIE_CATEGORIES), chart.y
        if chart.chart_type == "bar" and plot_df[x_col].nunique() > MAX_BAR_CATEGORIES:
            limited = plot_df.groupby(x_col, dropna=False)[chart.y].sum().reset_index()
            return _limit_categories(limited, x_col, chart.y, MAX_BAR_CATEGORIES), chart.y
        return plot_df, chart.y

    return None, None


def render_chart(df, chart: ChartSpec):
    plot_df, y_col = prepare_plot_data(df, chart)
    if plot_df is None or y_col is None:
        return None

    x_col = PERIOD_COLUMN if PERIOD_COLUMN in plot_df.columns else chart.x

    if chart.chart_type == "line":
        fig = px.line(plot_df, x=x_col, y=y_col, title=chart.title, markers=True)
        _apply_line_layout(fig, x_col, plot_df)

    elif chart.chart_type == "bar":
        fig = px.bar(plot_df, x=x_col, y=y_col, title=chart.title)

    elif chart.chart_type == "scatter":
        fig = px.scatter(plot_df, x=x_col, y=y_col, title=chart.title)

    elif chart.chart_type == "pie":
        fig = px.pie(plot_df, names=x_col, values=y_col, title=chart.title)

    else:
        return None

    return fig
