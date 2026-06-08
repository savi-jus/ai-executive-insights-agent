"""Pydantic models for structured chart recommendations from the LLM."""

from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

# Allowed values enforced on parsed LLM output
Aggregation = Literal["none", "count", "sum", "mean"]
TimeFreq = Literal["D", "W", "ME", "QE", "YE"]
ChartType = Literal["line", "bar", "pie", "scatter", "heatmap"]
BarMode = Literal["group", "stack"]


class ChartSpec(BaseModel):
    """Single chart definition: axes, aggregation, and executive-facing metadata."""

    title: str
    chart_type: ChartType
    x: str
    y: Optional[str] = None
    reason: str
    aggregation: Aggregation = "none"
    time_freq: Optional[TimeFreq] = None
    log_y: bool = False
    group_by: Optional[str] = None
    bar_mode: Optional[BarMode] = None
    time_column: Optional[str] = None

    @field_validator("y", mode="before")
    @classmethod
    def empty_y_to_none(cls, value):
        """Treat blank y values from the LLM as missing (e.g. count charts)."""
        if value is None or value == "":
            return None
        return value

    @field_validator("group_by", "time_column", mode="before")
    @classmethod
    def empty_optional_str_to_none(cls, value):
        if value is None or value == "":
            return None
        return value

    @field_validator("chart_type", mode="before")
    @classmethod
    def normalize_chart_type(cls, value):
        """Normalize casing so 'Line' and 'line' both validate."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("bar_mode", mode="before")
    @classmethod
    def normalize_bar_mode(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "clustered": "group",
                "cluster": "group",
                "grouped": "group",
                "side_by_side": "group",
                "stacked": "stack",
            }
            return aliases.get(normalized, normalized)
        return value

    @field_validator("time_freq", mode="before")
    @classmethod
    def normalize_time_freq(cls, value):
        """Map legacy pandas freq codes (M/Q/Y) to current ones (ME/QE/YE)."""
        if value is None:
            return None
        legacy = {"M": "ME", "Q": "QE", "Y": "YE"}
        return legacy.get(value, value)


class ChartRecommendations(BaseModel):
    """Top-level response wrapper: up to four ChartSpec objects."""

    charts: List[ChartSpec]
