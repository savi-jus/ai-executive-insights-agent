import json
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

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

PERIOD_COLUMN = "_period"


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
    normalized = column_name.strip().replace(" ", "")
    return bool(ID_COLUMN_PATTERN.search(normalized) or ID_COLUMN_PATTERN.search(column_name))


def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def find_year_month_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    year_col = month_col = None
    for col in df.columns:
        name = col.lower().strip()
        if name in ("year", "yr") or name.endswith(" year"):
            year_col = col
        if name in ("month", "mo", "mnth") or name.endswith(" month"):
            month_col = col
    return year_col, month_col


def is_year_like_column(column: str, series: pd.Series) -> bool:
    if "year" in column.lower():
        return True
    if is_numeric_series(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return non_null.min() >= 1900 and non_null.max() <= 2100 and non_null.nunique() <= 40
    return False


def is_month_like_column(column: str) -> bool:
    name = column.lower().strip()
    return name in ("month", "mo", "mnth") or name.endswith(" month")


def build_period_series(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    years = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    months = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    labels = years.astype(str) + "-" + months.astype(str).str.zfill(2) + "-01"
    return pd.to_datetime(labels, errors="coerce")


def title_implies_metric_trend(title: str) -> bool:
    return bool(METRIC_TREND_TITLE_PATTERN.search(title))


def title_implies_headcount_trend(title: str) -> bool:
    return bool(HEADCOUNT_TITLE_PATTERN.search(title))


def find_metric_column_for_title(df: pd.DataFrame, title: str) -> str | None:
    title_lower = title.lower()
    year_col, month_col = find_year_month_columns(df)
    skip = {c for c in (year_col, month_col) if c}

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
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        parsed = pd.to_datetime(series.head(20), errors="coerce")
        return parsed.notna().mean() >= 0.8
    return False


def column_metadata(df: pd.DataFrame, column: str) -> dict:
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

    schema = {
        "rows": int(df.shape[0]),
        "columns": columns,
        "hints": hints,
        "sample_rows": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
    return to_json_safe(schema)
