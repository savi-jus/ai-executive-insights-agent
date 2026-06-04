from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

Aggregation = Literal["none", "count", "sum", "mean"]
TimeFreq = Literal["D", "W", "ME", "QE", "YE"]
ChartType = Literal["line", "bar", "pie", "scatter"]


class ChartSpec(BaseModel):
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
        if value is None or value == "":
            return None
        return value

    @field_validator("chart_type", mode="before")
    @classmethod
    def normalize_chart_type(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("time_freq", mode="before")
    @classmethod
    def normalize_time_freq(cls, value):
        if value is None:
            return None
        legacy = {"M": "ME", "Q": "QE", "Y": "YE"}
        return legacy.get(value, value)


class ChartRecommendations(BaseModel):
    charts: List[ChartSpec]
