"""Turn validated ChartSpec objects into Plotly figures."""

import pandas as pd
import plotly.express as px

from tools.chart_schema import (
    MAX_PIE_CATEGORIES,
    PERIOD_COLUMN,
    build_period_series,
    chart_display_title,
    correlation_metric_columns,
    filter_df_to_time_period,
    find_year_month_columns,
    is_datetime_series,
    is_month_like_column,
    is_numeric_series,
    is_year_like_column,
    looks_like_rating_column,
    resolve_timeline_series,
    values_suggest_log_scale,
)
from tools.charts_model import ChartSpec

MAX_BAR_CATEGORIES = 20
COUNT_COLUMN = "count"
DISTRIBUTION_BIN_COUNT = 12

# Older pandas freq aliases still accepted from LLM output
LEGACY_FREQ_MAP = {"M": "ME", "Q": "QE", "Y": "YE"}


def resolve_time_freq(freq: str | None) -> str:
    """Default to monthly (ME) and normalize legacy freq codes."""
    raw = freq or "ME"
    return LEGACY_FREQ_MAP.get(raw, raw)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime when it is not already typed as such."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _filter_df_by_time_period(
    df: pd.DataFrame,
    chart: ChartSpec,
    filter_period: pd.Timestamp | None,
    filter_granularity: str | None,
) -> pd.DataFrame:
    """Keep rows for the selected time bucket when the period slider is active."""
    if filter_period is None or not filter_granularity:
        return df

    timeline = resolve_timeline_series(df, chart)
    if timeline is None:
        return df

    return filter_df_to_time_period(df, timeline, filter_period, filter_granularity)


def _limit_categories(df: pd.DataFrame, category_col: str, value_col: str, limit: int) -> pd.DataFrame:
    """Keep top N categories by value sum and bucket the rest as 'Other'."""
    ranked = df.groupby(category_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    top = ranked.head(limit).index.tolist()
    plot_df = df.copy()
    # Keep one dtype so sorting/plotting does not mix ints and strings (e.g. 2021 vs "Other")
    plot_df[category_col] = plot_df[category_col].astype(str)
    top = [str(value) for value in top]
    plot_df[category_col] = plot_df[category_col].where(plot_df[category_col].isin(top), "Other")
    return plot_df.groupby(category_col, dropna=False)[value_col].sum().reset_index()


def _needs_numeric_binning(x_col: str, series: pd.Series) -> bool:
    """True when x is a continuous numeric field with too many distinct values to plot raw."""
    if not is_numeric_series(series):
        return False
    if is_year_like_column(x_col, series) or is_month_like_column(x_col):
        return False
    if is_datetime_series(series):
        return False
    if looks_like_rating_column(x_col, series):
        return False
    return series.nunique(dropna=True) > MAX_PIE_CATEGORIES


def _apply_numeric_bins(plot_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Bucket continuous numeric x values into readable histogram bins."""
    numeric = pd.to_numeric(plot_df[x_col], errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return plot_df

    span = valid.max() - valid.min()
    if span <= 0:
        return plot_df

    bin_count = min(DISTRIBUTION_BIN_COUNT, max(5, int(valid.nunique() ** 0.5)))
    binned = pd.cut(numeric, bins=bin_count, duplicates="drop")
    result = plot_df.copy()
    result[x_col] = binned.astype(str)
    return result.dropna(subset=[x_col])


def _apply_bar_category_layout(fig, x_col: str, plot_df: pd.DataFrame) -> None:
    """Use categorical x labels when bins or long category names would overlap."""
    if x_col not in plot_df.columns:
        return
    labels = plot_df[x_col].astype(str)
    if labels.str.contains(r"[\(\[]", regex=True).any() or labels.str.len().max() > 8:
        fig.update_xaxes(type="category", tickangle=-35)


def _group_keys(x_col: str, chart: ChartSpec) -> list[str]:
    if chart.group_by and chart.group_by != x_col:
        return [x_col, chart.group_by]
    return [x_col]


def _should_use_year_month_period(df: pd.DataFrame, chart: ChartSpec) -> bool:
    """True when Year+Month columns should be merged into a single time axis."""
    year_col, month_col = find_year_month_columns(df)
    if not year_col or not month_col:
        return False
    if chart.x not in (year_col, month_col):
        return False
    if chart.chart_type not in ("line", "bar"):
        return False
    return True


def _resolve_x_column(df: pd.DataFrame, chart: ChartSpec) -> tuple[pd.DataFrame, str]:
    """Return working copy of df and the x column name to plot (may be synthetic _period)."""
    plot_df = df.copy()
    if _should_use_year_month_period(plot_df, chart):
        year_col, month_col = find_year_month_columns(plot_df)
        plot_df[PERIOD_COLUMN] = build_period_series(plot_df, year_col, month_col)
        plot_df = plot_df.dropna(subset=[PERIOD_COLUMN])
        return plot_df, PERIOD_COLUMN

    if chart.x in plot_df.columns and is_year_like_column(chart.x, plot_df[chart.x]):
        # Plot years as categorical strings so axes don't show decimals
        plot_df[chart.x] = pd.to_numeric(plot_df[chart.x], errors="coerce").astype("Int64").astype(str)

    return plot_df, chart.x


def _sort_plot_df(plot_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Sort x chronologically for dates, numerically for year-like values, else as text."""
    if x_col not in plot_df.columns:
        return plot_df

    series = plot_df[x_col]
    if x_col == PERIOD_COLUMN or pd.api.types.is_datetime64_any_dtype(series):
        return plot_df.sort_values(x_col)

    if pd.api.types.is_numeric_dtype(series):
        return plot_df.sort_values(x_col)

    parsed_num = pd.to_numeric(series, errors="coerce")
    if parsed_num.notna().mean() >= 0.8:
        return (
            plot_df.assign(_sort_key=parsed_num)
            .sort_values("_sort_key")
            .drop(columns="_sort_key")
        )

    if is_datetime_series(series):
        parsed_dt = pd.to_datetime(series, errors="coerce")
        if parsed_dt.notna().mean() >= 0.8:
            return plot_df.assign(**{x_col: parsed_dt}).sort_values(x_col)

    return plot_df.sort_values(x_col, key=lambda s: s.astype(str))


def _apply_scatter_layout(fig, x_col: str, y_col: str, plot_df: pd.DataFrame) -> None:
    """Keep numeric axes linear for continuous scatter plots."""
    if x_col in plot_df.columns and is_numeric_series(plot_df[x_col]):
        fig.update_xaxes(type="linear")
    if y_col in plot_df.columns and is_numeric_series(plot_df[y_col]):
        fig.update_yaxes(type="linear")


def _apply_line_layout(fig, x_col: str, plot_df: pd.DataFrame) -> None:
    """Format line chart x-axis for monthly periods vs year categories."""
    if x_col == PERIOD_COLUMN or is_datetime_series(plot_df[x_col]):
        fig.update_xaxes(tickformat="%b %Y", title="Month")
    elif x_col in plot_df.columns and is_year_like_column(x_col, plot_df[x_col]):
        fig.update_xaxes(type="category", title=x_col)


def _is_datetime_x_axis(x_col: str, plot_df: pd.DataFrame) -> bool:
    if x_col not in plot_df.columns:
        return False
    if x_col == PERIOD_COLUMN:
        return True
    return is_datetime_series(plot_df[x_col]) or pd.api.types.is_datetime64_any_dtype(plot_df[x_col])


def _apply_time_series_range_slider(fig, x_col: str, plot_df: pd.DataFrame, chart: ChartSpec) -> None:
    """Add an x-axis range slider when the trend has many datetime points."""
    if chart.chart_type not in ("line", "bar") or not _is_datetime_x_axis(x_col, plot_df):
        return

    x_values = pd.to_datetime(plot_df[x_col], errors="coerce").dropna().sort_values()
    if len(x_values) < 8:
        return

    min_x, max_x = x_values.min(), x_values.max()
    span = max_x - min_x
    if span <= pd.Timedelta(0):
        return

    # Default view: most recent ~40% of the timeline for readability
    view_start = max(min_x, max_x - span * 0.4)
    fig.update_xaxes(
        range=[view_start, max_x],
        rangeslider=dict(visible=True),
        tickformat="%b %Y",
    )


def _apply_legend_layout(fig, chart: ChartSpec) -> None:
    """Enable click-to-toggle series in the legend."""
    if not chart.group_by:
        return
    fig.update_layout(
        legend=dict(
            title_text=chart.group_by,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        )
    )


def _apply_log_y_axis(fig, plot_df: pd.DataFrame, y_col: str, chart: ChartSpec) -> None:
    """Use a log y-axis when requested or when aggregated values span a wide range."""
    if chart.chart_type not in ("line", "bar", "scatter"):
        return
    if y_col not in plot_df.columns:
        return

    use_log = chart.log_y or values_suggest_log_scale(plot_df[y_col])
    if not use_log:
        return

    values = plot_df[y_col].dropna()
    if values.empty or (values <= 0).any():
        return

    fig.update_yaxes(type="log")


def prepare_plot_data(df: pd.DataFrame, chart: ChartSpec) -> tuple[pd.DataFrame | None, str | None]:
    """Aggregate and reshape df according to chart spec; returns (plot_df, y_column)."""
    if chart.x not in df.columns:
        return None, None

    plot_df, x_col = _resolve_x_column(df, chart)
    agg = chart.aggregation
    freq = resolve_time_freq(chart.time_freq)
    group_keys = _group_keys(x_col, chart)

    if (
        not chart.group_by
        and chart.chart_type in ("bar", "pie")
        and agg in ("count", "none")
        and _needs_numeric_binning(x_col, plot_df[x_col])
    ):
        plot_df = _apply_numeric_bins(plot_df, x_col)
        if agg == "none":
            agg = "count"

    if agg == "count":
        x_series = plot_df[x_col]
        if is_datetime_series(x_series) or x_col == PERIOD_COLUMN:
            if x_col != PERIOD_COLUMN:
                plot_df[x_col] = _parse_datetime(plot_df[x_col])
            if chart.group_by and chart.group_by in plot_df.columns:
                grouped = (
                    plot_df.groupby([pd.Grouper(key=x_col, freq=freq), chart.group_by])
                    .size()
                    .reset_index(name=COUNT_COLUMN)
                )
            else:
                grouped = (
                    plot_df.groupby(pd.Grouper(key=x_col, freq=freq))
                    .size()
                    .reset_index(name=COUNT_COLUMN)
                )
            return _sort_plot_df(grouped.dropna(subset=[x_col]), x_col), COUNT_COLUMN

        if chart.group_by and chart.group_by in plot_df.columns:
            grouped = (
                plot_df.groupby(group_keys, dropna=False).size().reset_index(name=COUNT_COLUMN)
            )
        else:
            grouped = plot_df.groupby(x_col, dropna=False).size().reset_index(name=COUNT_COLUMN)

        if chart.chart_type == "bar" and not chart.group_by and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, COUNT_COLUMN, MAX_BAR_CATEGORIES)
        if chart.chart_type == "pie" and len(grouped) > MAX_PIE_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, COUNT_COLUMN, MAX_PIE_CATEGORIES)
        return _sort_plot_df(grouped, x_col), COUNT_COLUMN

    if agg in ("sum", "mean") and chart.y and chart.y in plot_df.columns:
        x_series = plot_df[x_col]
        if is_datetime_series(x_series) or x_col == PERIOD_COLUMN:
            if x_col != PERIOD_COLUMN:
                plot_df[x_col] = _parse_datetime(plot_df[x_col])
            if chart.group_by and chart.group_by in plot_df.columns:
                grouped = plot_df.groupby([pd.Grouper(key=x_col, freq=freq), chart.group_by])[chart.y]
            else:
                grouped = plot_df.groupby(pd.Grouper(key=x_col, freq=freq))[chart.y]
            if agg == "sum":
                result = grouped.sum().reset_index()
            else:
                result = grouped.mean().reset_index()
            return _sort_plot_df(result.dropna(subset=[x_col]), x_col), chart.y

        if chart.group_by and chart.group_by in plot_df.columns:
            grouped = plot_df.groupby(group_keys, dropna=False)[chart.y]
        else:
            grouped = plot_df.groupby(x_col, dropna=False)[chart.y]

        if agg == "sum":
            grouped = grouped.sum().reset_index()
        else:
            grouped = grouped.mean().reset_index()

        if chart.chart_type == "bar" and not chart.group_by and len(grouped) > MAX_BAR_CATEGORIES:
            grouped = _limit_categories(grouped, x_col, chart.y, MAX_BAR_CATEGORIES)
        return _sort_plot_df(grouped, x_col), chart.y

    # aggregation == "none": plot raw rows (with category limits for pie/bar)
    if chart.y and chart.y in plot_df.columns:
        if is_datetime_series(plot_df[x_col]):
            plot_df[x_col] = _parse_datetime(plot_df[x_col])
        plot_df = _sort_plot_df(plot_df, x_col)
        if chart.chart_type == "pie" and plot_df[x_col].nunique() > MAX_PIE_CATEGORIES:
            limited = plot_df.groupby(x_col, dropna=False)[chart.y].sum().reset_index()
            return _limit_categories(limited, x_col, chart.y, MAX_PIE_CATEGORIES), chart.y
        if chart.chart_type == "bar" and not chart.group_by and plot_df[x_col].nunique() > MAX_BAR_CATEGORIES:
            limited = plot_df.groupby(x_col, dropna=False)[chart.y].sum().reset_index()
            return _limit_categories(limited, x_col, chart.y, MAX_BAR_CATEGORIES), chart.y
        return plot_df, chart.y

    return None, None


def render_chart(
    df,
    chart: ChartSpec,
    *,
    filter_period: pd.Timestamp | None = None,
    filter_granularity: str | None = None,
):
    """Build a Plotly figure from prepared data, or None if preparation failed."""
    plot_df = _filter_df_by_time_period(df, chart, filter_period, filter_granularity)
    if plot_df.empty:
        return None

    plot_title = "" #chart_display_title(chart)

    if chart.chart_type == "heatmap":
        metric_cols = correlation_metric_columns(plot_df)
        if len(metric_cols) < 2:
            return None
        corr = plot_df[metric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title=plot_title,
        )
        fig.update_layout(xaxis_side="top")
        return fig

    plot_df, y_col = prepare_plot_data(plot_df, chart)
    if plot_df is None or y_col is None:
        return None

    x_col = PERIOD_COLUMN if PERIOD_COLUMN in plot_df.columns else chart.x
    color_col = chart.group_by if chart.group_by and chart.group_by in plot_df.columns else None

    if chart.chart_type == "line":
        fig = px.line(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=plot_title,
            markers=True,
        )
        _apply_line_layout(fig, x_col, plot_df)
        _apply_time_series_range_slider(fig, x_col, plot_df, chart)
        _apply_legend_layout(fig, chart)
        _apply_log_y_axis(fig, plot_df, y_col, chart)

    elif chart.chart_type == "bar":
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            barmode=chart.bar_mode if color_col else "relative",
            title=plot_title,
        )
        _apply_time_series_range_slider(fig, x_col, plot_df, chart)
        _apply_bar_category_layout(fig, x_col, plot_df)
        _apply_legend_layout(fig, chart)
        _apply_log_y_axis(fig, plot_df, y_col, chart)

    elif chart.chart_type == "scatter":
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=plot_title,
        )
        _apply_scatter_layout(fig, x_col, y_col, plot_df)
        _apply_legend_layout(fig, chart)
        _apply_log_y_axis(fig, plot_df, y_col, chart)

    elif chart.chart_type == "pie":
        fig = px.pie(plot_df, names=x_col, values=y_col, title=plot_title)

    else:
        return None

    return fig
