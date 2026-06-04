import json
import re

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


def looks_like_id(column_name: str) -> bool:
    normalized = column_name.strip().replace(" ", "")
    return bool(ID_COLUMN_PATTERN.search(normalized) or ID_COLUMN_PATTERN.search(column_name))


def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


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
        "is_numeric": is_numeric_series(series),
        "is_datetime": is_datetime_series(series),
        "looks_like_id": looks_like_id(column),
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
    elif nunique <= 15:
        metadata["top_values"] = series.value_counts(dropna=True).head(5).to_dict()
    return metadata


def build_chart_schema(df: pd.DataFrame) -> dict:
    columns = [column_metadata(df, col) for col in df.columns]
    return {
        "rows": int(df.shape[0]),
        "columns": columns,
        "sample_rows": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
