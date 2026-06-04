"""Pydantic models for structured chart recommendations from the LLM."""

from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

# Allowed values enforced on parsed LLM output
Aggregation = Literal["none", "count", "sum", "mean"]
TimeFreq = Literal["D", "W", "ME", "QE", "YE"]
ChartType = Literal["line", "bar", "pie", "scatter"]


class ChartSpec(BaseModel):
    """Single chart definition: axes, aggregation, and executive-facing metadata."""

    title: str
    chart_type: ChartType
    x: str
    y: Optional[str] = None
    reason: str
    aggregation: Aggregation = "none"
    time_freq: Optional[TimeFreq] = None

    @field_validator("y", mode="before")
    @classmethod
    def empty_y_to_none(cls, value):
        """Treat blank y values from the LLM as missing (e.g. count charts)."""
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

    @field_validator("time_freq", mode="before")
    @classmethod
    def normalize_time_freq(cls, value):
        """Map legacy pandas freq codes (M/Q/Y) to current ones (ME/QE/YE)."""
        if value is None:
            return None
        legacy = {"M": "ME", "Q": "QE", "Y": "YE"}
        return legacy.get(value, value)


class ChartRecommendations(BaseModel):
    """Top-level response wrapper: up to three ChartSpec objects."""

    charts: List[ChartSpec]
